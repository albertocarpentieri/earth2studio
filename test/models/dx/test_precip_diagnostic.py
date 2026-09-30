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
from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.models.auto import Package
from earth2studio.models.dx import PrecipitationDiagnostic
from earth2studio.utils import handshake_dim

VARIABLES = ["t2m", "tcwv", "q500"]
NLAT, NLON = 16, 32


class PhooBackbone(torch.nn.Module):
    """Spoof core backbone: returns a single output channel."""

    def forward(self, x, noise=None):
        return x[:, :1, :, :]


def _grid(nlat, nlon):
    return (
        np.linspace(90, -90, nlat, endpoint=True),
        np.linspace(0, 360, nlon, endpoint=False),
    )


def _make_model(device, state_num_steps=2, model_shape=(NLAT, NLON)):
    n_in = len(VARIABLES) * state_num_steps
    native_lat, native_lon = _grid(NLAT, NLON)
    model_lat, model_lon = _grid(*model_shape)
    model = PrecipitationDiagnostic(
        PhooBackbone(),
        center=torch.zeros(1, n_in, 1, 1),
        scale=torch.ones(1, n_in, 1, 1),
        target_scale=torch.ones(1, 1, 1, 1),
        invariants=torch.zeros(1, 6, *model_shape),
        variables=VARIABLES,
        native_lat=native_lat,
        native_lon=native_lon,
        model_lat=model_lat,
        model_lon=model_lon,
        state_num_steps=state_num_steps,
    )
    return model.to(device)


def _coords(dx, n_batch):
    return OrderedDict(
        {
            "batch": np.ones(n_batch),
            "time": np.array([np.datetime64("2023-01-01T00:00")]),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )


@pytest.mark.parametrize("n_batch", [1, 2])
@pytest.mark.parametrize("state_num_steps", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_precip_diagnostic(n_batch, state_num_steps, device):
    dx = _make_model(device, state_num_steps=state_num_steps)
    n_var = len(VARIABLES) * state_num_steps
    # Frame-major variable axis: each variable once per input frame.
    assert len(dx.input_coords()["variable"]) == n_var
    x = torch.randn(n_batch, 1, 1, n_var, NLAT, NLON).to(device)
    coords = _coords(dx, n_batch)

    out, out_coords = dx(x, coords)

    assert out.shape == torch.Size([n_batch, 1, 1, 1, NLAT, NLON])
    assert out_coords["variable"] == dx.output_coords(coords)["variable"]
    # Precipitation is non-negative in metres.
    assert torch.all(out >= 0.0)
    handshake_dim(out_coords, "lon", 5)
    handshake_dim(out_coords, "lat", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_precip_diagnostic_downsample(device):
    model_shape = (NLAT // 2 + 1, NLON // 2)
    dx = _make_model(device, state_num_steps=1, model_shape=model_shape)
    assert dx.resampler is not None
    x = torch.randn(1, 1, 1, len(VARIABLES), NLAT, NLON).to(device)

    out, out_coords = dx(x, _coords(dx, 1))

    assert out.shape == torch.Size([1, 1, 1, 1, *model_shape])
    assert out_coords["lat"].shape == (model_shape[0],)
    assert out_coords["lon"].shape == (model_shape[1],)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_precip_diagnostic_exceptions(device):
    dx = _make_model(device, state_num_steps=2)
    n_var = len(VARIABLES) * 2
    x = torch.randn(1, 1, 1, n_var, NLAT, NLON).to(device)

    # Wrong dimension name.
    wrong = _coords(dx, 1)
    wrong = OrderedDict(
        {("wrong" if k == "variable" else k): v for k, v in wrong.items()}
    )
    with pytest.raises((KeyError, ValueError)):
        dx(x, wrong)

    # Swapped lat/lon ordering.
    swapped = OrderedDict(
        {
            "batch": np.ones(1),
            "time": np.array([np.datetime64("2023-01-01T00:00")]),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": dx.input_coords()["variable"],
            "lon": dx.input_coords()["lon"],
            "lat": dx.input_coords()["lat"],
        }
    )
    with pytest.raises(ValueError):
        dx(x, swapped)


def test_precip_diagnostic_default_package_requires_env(monkeypatch):
    monkeypatch.delenv("PRECIP_DIAGNOSTIC_PACKAGE", raising=False)
    with pytest.raises(ValueError):
        PrecipitationDiagnostic.load_default_package()


def test_precip_diagnostic_load_model(tmp_path):
    s2t = pytest.importorskip("physicsnemo.experimental.models.s2transformer")
    state_num_steps = 2
    n_in = len(VARIABLES) * state_num_steps
    core = s2t.S2Transformer(
        model_grid_type="equiangular",
        sht_grid_type="equiangular",
        inp_shape=(NLAT, NLON),
        out_shape=(NLAT, NLON),
        in_channels=n_in + 6,
        out_channels=1,
        scale_factor=1,
        processor={
            "embed_dim": 8,
            "num_layers": 1,
            "num_heads": 1,
            "attention_mode": "global",
            "attn_optimized_kernel": False,
        },
        pos_embed={"kind": "spectral", "application": "qk"},
    )
    core.save(str(tmp_path / "core.mdlus"))

    lat, lon = _grid(NLAT, NLON)
    np.save(tmp_path / "center.npy", np.zeros(n_in, dtype=np.float32))
    np.save(tmp_path / "scale.npy", np.ones(n_in, dtype=np.float32))
    np.save(tmp_path / "target_scale.npy", np.ones(1, dtype=np.float32))
    np.save(tmp_path / "invariants.npy", np.zeros((6, NLAT, NLON), dtype=np.float32))
    with open(tmp_path / "config.json", "w") as f:
        json.dump(
            {
                "variables": VARIABLES,
                "state_num_steps": state_num_steps,
                "native_lat": lat.tolist(),
                "native_lon": lon.tolist(),
                "model_lat": lat.tolist(),
                "model_lon": lon.tolist(),
                "precip_scale": 1.0e-3,
                "out_activation": "softclamp",
            },
            f,
        )

    dx = PrecipitationDiagnostic.load_model(Package(str(tmp_path)))
    assert dx.state_num_steps == state_num_steps
    assert dx.resampler is None

    x = torch.randn(1, 1, 1, n_in, NLAT, NLON)
    out, out_coords = dx(x, _coords(dx, 1))
    assert out.shape == torch.Size([1, 1, 1, 1, NLAT, NLON])
    assert torch.all(torch.isfinite(out))
    assert torch.all(out >= 0.0)
    assert list(out_coords["variable"]) == ["tp06"]
