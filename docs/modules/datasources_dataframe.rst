.. _earth2studio.data.dataframe:

:mod:`earth2studio.data`: DataFrame Sources
--------------------------------------------

Data sources that provide tabular data as DataFrames.

.. currentmodule:: earth2studio

.. badge-filter:: region:global region:na region:eu region:as region:au region:af region:sa
   dataclass:analysis dataclass:reanalysis dataclass:observation dataclass:simulation
   product:wind product:precip product:temp product:atmos product:ocean product:land product:veg product:solar product:radar product:sat product:insitu
   :filter-mode: or
   :badge-order-fixed:
   :group-visibility-toggle:
   :group-hidden: product

   .. autosummary::
      :nosignatures:
      :toctree: generated/data/
      :template: datasource.rst

      data.GHCNDaily
      data.GOESGLM
      data.ISD
      data.JPSS_ATMS
      data.JPSS_CRIS
      data.MetOpAMSUA
      data.MetOpAVHRR
      data.MetOpIASI
      data.MetOpMHS
      data.NNJAObsConv
<<<<<<< HEAD
<<<<<<< HEAD
      data.NNJAObsSat
=======
>>>>>>> 3e27b0e34a010228ea22e466bab9a8cd66ec4dc8
=======
      data.NomadsGDASObsConv
>>>>>>> 24696d5901f92adc6346b436a8f94db0792ce11e
      data.RandomDataFrame
      data.UFSObsConv
      data.UFSObsSat
