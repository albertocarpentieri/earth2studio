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

from collections.abc import Callable

import numpy as np
import pandas as pd

from .base import LexiconType


class NNJAObsConvLexicon(metaclass=LexiconType):
    """NOAA-NASA Joint Archive (NNJA) PrepBUFR lexicon for conventional
    (in-situ) observations.

    Maps Earth2Studio variable names to PrepBUFR mnemonics in the NNJA
    ``conv/prepbufr/`` archive. Vocab values are either a PrepBUFR
    mnemonic (e.g. ``"TOB"`` for temperature) or ``"wind::u"`` /
    ``"wind::v"`` for wind components decomposed from the UOB/VOB level
    fields.

    Modifier functions convert raw PrepBUFR observation values to
    Earth2Studio standard units:

    - ``t``: TOB (DEG C) -> Kelvin (+273.15)
    - ``q``: QOB (mg/kg) -> kg kg-1 (/1e6)
    - ``pres``: POB (hPa / MB) -> Pa (*100)
    - ``u``, ``v``: UOB/VOB already in m s-1 (no conversion)

    Note
    ----
    This lexicon parallels :py:class:`earth2studio.lexicon.GDASObsConvLexicon`
    but is kept separate to keep the NNJA data source self-contained.
    A future refactor may merge them into a shared PrepBUFR vocabulary.

    Additional resources on PrepBUFR format and observation types:

    - https://psl.noaa.gov/data/nnja_obs/
    - https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/table_2.htm
    - https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/document.htm
    """

    VOCAB: dict[str, str] = {
        "u": "wind::u",
        "v": "wind::v",
        "q": "QOB",
        "t": "TOB",
        "pres": "POB",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[..., pd.DataFrame]]:
        """Get item from PrepBUFR vocabulary.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            - PrepBUFR vocab string (mnemonic or ``"wind::u"``/``"wind::v"``)
            - A modifier function applied to the loaded DataFrame to convert
              the ``observation`` column from raw PrepBUFR units to
              Earth2Studio standard units.
        """
        bufr_key = cls.VOCAB[val]

        if val == "t":

            def mod(df: pd.DataFrame) -> pd.DataFrame:
                df["observation"] = np.float32(df["observation"] + 273.15)
                return df

        elif val == "q":

            def mod(df: pd.DataFrame) -> pd.DataFrame:
                df["observation"] = np.float32(df["observation"] * 1e-6)
                return df

        elif val == "pres":

            def mod(df: pd.DataFrame) -> pd.DataFrame:
                df["observation"] = np.float32(df["observation"] * 100.0)
                return df

        else:

            def mod(df: pd.DataFrame) -> pd.DataFrame:
                return df

        return bufr_key, mod


class NNJASatelliteLexicon(metaclass=LexiconType):
    """NOAA-NASA Joint Archive (NNJA) lexicon for satellite-radiance BUFR
    observations.

    Maps Earth2Studio variable names to ``(sensor, source, platforms,
    bufr_key)`` strings used by :py:class:`earth2studio.data.NNJAObsSat`
    to template S3 paths and decode WMO-standard satellite BUFR files.

    The vocab encoding is::

        "<sensor>::<source>::<comma-separated platforms>::<bufr_key>"

    where ``<sensor>`` and ``<source>`` are the first two NNJA bucket
    path segments (``noaa-reanalyses-pds/observations/reanalysis/{sensor}/{source}/...``),
    ``<platforms>`` is the comma-separated list of satellite platforms
    that this sensor/source folder carries (used to filter when the user
    restricts ``satellites=...`` at construction time), and
    ``<bufr_key>`` is the eccodes BUFR key holding the radiance
    observation (e.g. ``brightnessTemperature``, ``scaledIasiRadiance``).

    Note
    ----
    Coverage starts narrow and grows as variables are tested. To request
    a missing sensor please open an issue.

    Additional resources:

    - https://psl.noaa.gov/data/nnja_obs/
    - https://registry.opendata.aws/noaa-reanalyses-obs/
    """

    VOCAB: dict[str, str] = {
        "atms": "atms::atms::npp,n20::brightnessTemperature",
        "amsua": "amsua::1bamua::n15,n16,n17,n18,n19,metop-a,metop-b,metop-c::brightnessTemperature",
        "amsua_aqua": "amsua::nasa::aqua::brightnessTemperature",
        "amsub": "amsub::1bamub::n15,n16,n17::brightnessTemperature",
        "mhs": "mhs::1bmhs::n18,n19,metop-a,metop-b,metop-c::brightnessTemperature",
        "iasi": "iasi::mtiasi::metop-a,metop-b,metop-c::scaledIasiRadiance",
        "cris": "cris::cris::npp::radiance",
        "crisfsr": "cris::crisf4::npp,n20::radiance",
        "hirs": "hirs::1bhrs3::n15,n16,n17::brightnessTemperature",
        "saphir": "saphir::saphir::megha-tropiques::brightnessTemperature",
        "gmi": "gmi::gmi::gpm::brightnessTemperature",
        "airs": "airs::airsev::aqua::scaledRadiance",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[..., pd.DataFrame]]:
        """Get item from satellite BUFR vocabulary.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            - NNJA vocab string ``"<sensor>::<source>::<platforms>::<bufr_key>"``.
            - Identity modifier function (no DataFrame transformation).
        """
        nnja_key = cls.VOCAB[val]

        def mod(df: pd.DataFrame) -> pd.DataFrame:
            return df

        return nnja_key, mod
