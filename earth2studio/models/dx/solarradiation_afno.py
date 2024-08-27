# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.nn.afno_ssrd import SolarRadiationNet
from earth2studio.models.nn.afno_precip import PrecipNet
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.type import CoordSystem

VARIABLES = [
     "t2m", 
     "sp", 
     "tcwv", 
     "z50", 
     "z100", 
     "z300", 
     "z500", 
     "z850", 
     "z925", 
     "z1000",
     "t50", 
     "t100", 
     "t300", 
     "t500", 
     "t850", 
     "t925",
     "t1000",
     "q50", 
     "q100", 
     "q300", 
     "q500", 
     "q850", 
     "q925", 
     "q1000"
]

class SolarRadiationAFNO(torch.nn.Module, AutoModelMixin):
    """Soalr Radiation AFNO diagnostic model. Predicts the accumulated global surface solar
    radiation over 6 hours [Jm^-2]. The model uses 31 variables as input and outputs 
    one on a 0.25 degree lat-lon grid (south-pole excluding) [720 x 1440].

    Parameters
    ----------
    core_model : torch.nn.Module
        Core pytorch model
    mean : torch.Tensor
        Model mean normalization tensor of size [31,1,1]
    std : torch.Tensor
        Model standard deviation normalization tensor of size [31,1,1]
    landsea_mask : torch.Tensor
        Land sea mask
    orography : torch.Tensor
        Surface geopotential (orography) 
    latlon: torch.Tensor
        4 fields embedding location information (cos(lat), sin(lat), cos(lon), sin(lon))
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
        ssrd_mean: torch.Tensor,
        ssrd_std: torch.Tensor,
        orography: torch.Tensor,
        landsea_mask: torch.Tensor,
        latlon: torch.Tensor
    ):
        super().__init__()
        self.core_model = core_model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("orography", orography)
        self.register_buffer("landsea_mask", landsea_mask)
        self.register_buffer("latlon", latlon)

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 720, endpoint=False),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of diagnostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "lon", 3)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "variable", 1)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(["ssrd"])
        return output_coords

    def __str__(self) -> str:
        return "solarnet"
    
    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained precipation model package from Nvidia model registry"""
        return Package(
            "ngc://models/nvidia/modulus/modulus_diagnostics@v0.1",
            cache_options={
                "cache_storage": Package.default_cache("precip_afno"),
                "same_names": True,
            },
        )
    
    @classmethod
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic from package"""
        checkpoint_zip = Path(package.resolve("ssrd_afno.zip"))
        # Have to manually unzip here. Should not zip checkpoints in the future
        with zipfile.ZipFile(checkpoint_zip, "r") as zip_ref:
            zip_ref.extractall(checkpoint_zip.parent)

        model = PrecipNet.from_checkpoint(
            str(
                checkpoint_zip.parent
                / Path("precipitation_afno/ssrd_afno.mdlus")
            )
        )
        model.eval()

        input_mean = torch.Tensor(
            np.load(
                str(checkpoint_zip.parent / Path("precipitation_afno/global_means.npy"))
            )
        )
        
        input_std = torch.Tensor(
            np.load(
                str(checkpoint_zip.parent / Path("precipitation_afno/global_stds.npy"))
            )
        )
        
        input_z = torch.Tensor(
            np.load(
                str(checkpoint_zip.parent / Path("precipitation_afno/orography.npy"))
            )
        )
        input_z = (input_z - inpu_z.mean()) / input_z.std()
        input_z = input_z.expand(input_z.shape[:1] + (2,) + input_z.shape[2:])
        
        input_lsm = torch.Tensor(
            np.load(
                str(checkpoint_zip.parent / Path("precipitation_afno/land_sea_mask.npy"))
            )
        )
        input_lsm = input_lsm.expand(input_lsm.shape[:1] + (2,) + input_lsm.shape[2:])
        
        input_latlon = torch.Tensor(
            np.load(
                str(checkpoint_zip.parent / Path("precipitation_afno/latlon.npy"))
            )
        )
        input_latlon = input_latlon.expand(input_latlon.shape[:1] + (2,) + input_latlon.shape[2:])
        return cls(model, input_center, input_scale, input_z, input_lsm, input_latlon)

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(coords)
        x = (x - self.mean) / self.std
        x = torch.cat((x, self.latlon, self.orography, self.landsea_mask), dim=2)
        out = self.core_model(x)*self.ssrd_std + self.ssrd_mean
        return out, output_coords
