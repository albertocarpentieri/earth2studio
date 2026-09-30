# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from collections import OrderedDict

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    from physicsnemo import Module
    from torch_harmonics.resample import ResampleS2
except ImportError:
    OptionalDependencyFailure("precip-diagnostic")
    Module = None
    ResampleS2 = None


def _soft_clamp(x: torch.Tensor) -> torch.Tensor:
    """Smooth non-negativity activation used at training time.

    Mirrors ``precip_diagnostic.diagnostic.diagnostic_wrapper._soft_clamp``:
    a quadratic ramp for ``0 < x < 0.5`` that switches to the identity (minus a
    constant) for ``x >= 0.5`` so the gradient stays bounded.
    """
    y = torch.where(x > 0.0, x**2, torch.zeros_like(x))
    y = torch.where(x >= 0.5, x - 0.25, y)
    return y


# Must match precip_diagnostic.diagnostic.diagnostic_wrapper._soft_clamp_sublinear.
# knee=4 -> ~21 mm/6h; knee=4.5 -> ~57 mm/6h; knee=5 -> ~159 mm/6h.
_SUBLINEAR_KNEE = 4.5
_SUBLINEAR_K = 1.0


def _soft_clamp_sublinear(x: torch.Tensor) -> torch.Tensor:
    """Unbounded soft-clamp with a sqrt (sublinear) upper tail.

    Identical to ``_soft_clamp`` below the knee; above it the output grows like
    sqrt(x) (C1-continuous), so a log-space overshoot can no longer explode
    exponentially through ``expm1`` while remaining unbounded.
    """
    knee, k = _SUBLINEAR_KNEE, _SUBLINEAR_K
    y = torch.where(x > 0.0, x**2, torch.zeros_like(x))
    y = torch.where(x >= 0.5, x - 0.25, y)
    t = torch.clamp(x - knee, min=0.0)
    sub = (knee - 0.25) + (torch.sqrt(1.0 + 2.0 * k * t) - 1.0) / k
    return torch.where(x >= knee, sub, y)


OUT_ACTIVATIONS = {
    "softclamp": _soft_clamp,
    "softclamp_sublinear": _soft_clamp_sublinear,
    "relu": torch.nn.functional.relu,
    "none": lambda x: x,
}


@check_optional_dependencies()
class PrecipitationDiagnostic(torch.nn.Module, AutoModelMixin):
    """S2Transformer precipitation diagnostic model.

    Predicts the 6-hour accumulated total precipitation ``tp06`` (units: m) from
    ERA5 atmospheric state. Inference path:

    1. (Optionally) spherically downsample the native ERA5 input grid to the
       model's working grid (the trained model may operate at a coarser grid).
    2. Per-channel state normalization (z-score for most variables, min-max for
       humidity ``q*`` / ``tcwv`` channels for FCN3 compatibility), baked into
       the ``center`` / ``scale`` buffers at packaging time.
    3. Concatenate invariant channels ``[sin(lat), cos(lat), sin(lon),
       cos(lon), orography, land_sea_mask]``.
    4. Run the core backbone (PhysicsNeMo ``S2Transformer``), apply the
       soft-clamp activation, scale by the (log-space) target standard
       deviation, and invert the ``log1p`` precipitation transform to return
       precipitation in metres.

    The model is config-driven: ``load_model`` reads a ``config.json`` plus
    baked artifacts from the package, so it supports both single-frame and
    multi-step (e.g. 2-step ``[T, T+6h]``) variants, and either a native or a
    downsampled working grid. For an ``N``-step model the input variable axis
    lists every state variable once per frame, frame-major, as
    ``<variable>::t{step}`` (frame 0 first). Input is on the native grid; output
    is on the (possibly coarser) model grid.

    Note
    ----
    The core backbone requires a PhysicsNeMo build that provides
    ``physicsnemo.experimental.models.s2transformer``. There is no public
    package yet; load weights from a local package directory with
    ``PrecipitationDiagnostic.load_model(Package("/path/to/package"))``.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core backbone mapping normalized state + invariants to one channel
    center : torch.Tensor
        Per-channel state normalization center of size [1, C, 1, 1]
    scale : torch.Tensor
        Per-channel state normalization scale of size [1, C, 1, 1]
    target_scale : torch.Tensor
        Log-space target standard deviation of size [1, 1, 1, 1]
    invariants : torch.Tensor
        Invariant channels on the model grid of size [1, 6, H, W]
    variables : list[str]
        State variables of a single input frame
    native_lat : np.ndarray
        Latitudes of the input grid
    native_lon : np.ndarray
        Longitudes of the input grid
    model_lat : np.ndarray
        Latitudes of the model (output) grid
    model_lon : np.ndarray
        Longitudes of the model (output) grid
    state_num_steps : int, optional
        Number of input frames, by default 1
    precip_scale : float, optional
        Scale of the ``log1p`` precipitation transform in metres, by default 1e-5
    out_activation : str, optional
        Output activation, one of ``softclamp``, ``softclamp_sublinear``,
        ``relu``, ``none``, by default ``softclamp``
    resample_mode : str, optional
        ``ResampleS2`` mode used when input and model grids differ, by default
        ``bilinear-spherical``

    Badges
    ------
    region:global class:medium-range product:precip year:2026 gpu:40gb
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
        target_scale: torch.Tensor,
        invariants: torch.Tensor,
        variables: list[str],
        native_lat: np.ndarray,
        native_lon: np.ndarray,
        model_lat: np.ndarray,
        model_lon: np.ndarray,
        state_num_steps: int = 1,
        precip_scale: float = 1.0e-5,
        out_activation: str = "softclamp",
        resample_mode: str = "bilinear-spherical",
    ):
        super().__init__()
        self.core_model = core_model
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.register_buffer("target_scale", target_scale)
        self.register_buffer("invariants", invariants)
        self._variables = list(variables)
        self.state_num_steps = int(state_num_steps)
        self._native_lat = np.asarray(native_lat)
        self._native_lon = np.asarray(native_lon)
        self._model_lat = np.asarray(model_lat)
        self._model_lon = np.asarray(model_lon)
        self.precip_scale = float(precip_scale)
        if out_activation not in OUT_ACTIVATIONS:
            raise ValueError(f"Unsupported out_activation: {out_activation}")
        self._out_activation = OUT_ACTIVATIONS[out_activation]

        # Spherical resampler native -> model grid (no-op when grids match).
        self.resampler: torch.nn.Module | None = None
        if (self._native_lat.shape[0], self._native_lon.shape[0]) != (
            self._model_lat.shape[0],
            self._model_lon.shape[0],
        ):
            self.resampler = ResampleS2(
                nlat_in=self._native_lat.shape[0],
                nlon_in=self._native_lon.shape[0],
                nlat_out=self._model_lat.shape[0],
                nlon_out=self._model_lon.shape[0],
                grid_in="equiangular",
                grid_out="equiangular",
                mode=resample_mode,
            )

        # Variable axis: frame-major, each state variable once per input frame.
        self._input_variables = np.array(
            [
                f"{v}::t{step}"
                for step in range(self.state_num_steps)
                for v in self._variables
            ]
        )

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model (native ERA5 grid)."""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.empty(0),
                "variable": self._input_variables.copy(),
                "lat": self._native_lat.copy(),
                "lon": self._native_lon.copy(),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of diagnostic model (model working grid)."""
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "lon", 5)
        handshake_dim(input_coords, "lat", 4)
        handshake_dim(input_coords, "variable", 3)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(["tp06"])
        output_coords["lat"] = self._model_lat.copy()
        output_coords["lon"] = self._model_lon.copy()
        return output_coords

    def __str__(self) -> str:
        return "PrecipDiagnostic"

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default diagnostic package.

        There is no public remote package for this model yet. The package root
        is read from the ``PRECIP_DIAGNOSTIC_PACKAGE`` environment variable.
        """
        root = os.environ.get("PRECIP_DIAGNOSTIC_PACKAGE")
        if not root:
            raise ValueError(
                "PrecipitationDiagnostic has no public package yet. Set "
                "PRECIP_DIAGNOSTIC_PACKAGE to the local package directory or use "
                "PrecipitationDiagnostic.load_model(Package(path))."
            )
        return Package(
            root,
            cache_options={
                "cache_storage": Package.default_cache("precipitation_diagnostic"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic from package"""
        with open(package.resolve("config.json")) as f:
            config = json.load(f)

        core_model = Module.from_checkpoint(package.resolve("core.mdlus"))
        core_model.eval()

        def _load(name: str) -> torch.Tensor:
            return torch.as_tensor(
                np.load(package.resolve(name)), dtype=torch.float32
            )

        center = _load("center.npy").reshape(1, -1, 1, 1)
        scale = _load("scale.npy").reshape(1, -1, 1, 1)
        target_scale = _load("target_scale.npy").reshape(1, 1, 1, 1)
        invariants = _load("invariants.npy")
        if invariants.dim() == 3:
            invariants = invariants.unsqueeze(0)

        return cls(
            core_model,
            center,
            scale,
            target_scale,
            invariants,
            variables=config["variables"],
            native_lat=np.array(config["native_lat"]),
            native_lon=np.array(config["native_lon"]),
            model_lat=np.array(config["model_lat"]),
            model_lon=np.array(config["model_lon"]),
            state_num_steps=int(config.get("state_num_steps", 1)),
            precip_scale=float(config.get("precip_scale", 1.0e-5)),
            out_activation=config.get("out_activation", "softclamp"),
            resample_mode=config.get("resample_mode", "bilinear-spherical"),
        )

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(coords)

        # Flatten the leading (batch, time, lead_time) dims into one so the core
        # model and broadcasted invariants/stats operate on (N, C, H, W).
        lead_shape = x.shape[:-3]
        x = x.flatten(0, len(lead_shape) - 1)

        # Native -> model grid (spherical), matching the training preprocessing.
        if self.resampler is not None:
            x = self.resampler(x.float())

        x = (x - self.center) / self.scale
        inv = self.invariants.expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, inv], dim=1)

        out = self.core_model(x)
        out = self._out_activation(out)
        out = out * self.target_scale
        # Invert the log1p precipitation transform back to metres.
        out = torch.expm1(out) * self.precip_scale
        out = out.clamp_min(0.0)

        out = out.reshape(*lead_shape, 1, out.shape[-2], out.shape[-1])
        return out, output_coords
