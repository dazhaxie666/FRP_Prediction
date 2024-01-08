# FRP_Prediction
The content of this repository is a supplementary material to the article 'A Deep Learning Framework: Predicting Fire Radiative Power From the Combination of Polar-Orbiting and Geostationary Satellite Data During Wildfire Spread', which mainly includes two aspects: data and models.
## Data
- Oringinal Data
  - FRP Data
    - MODIS FRP 
      - [Terra FRP](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD14A1)
      - [Auqa FRP](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD14A1)
    - [GOES FRP](https://developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_16_FDCC)
  - Dynamic Data
    - [Min Temperature](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET#bands)
    - [Max Temperature](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET#bands)
    - [Precipitation](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET#bands)
    - [Wind Velocity](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET#bands)
    - [Wind Direction](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET#bands)
    - [Min Relative Humidity](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET#bands)
    - [Max Relative Humidity](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET#bands)
    - [100h Fuel Moisture](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET#bands)
    - [1000h Fuel Moisture](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET#bands)
  - Constant Data
    - [Elevation](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003)
    - [Aspect](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003)
    - [Slope](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003)
    - [Land cover](https://developers.google.com/earth-engine/datasets/catalog/USGS_NLCD_RELEASES_2020_REL_NALCMS)
    - [Population density](https://developers.google.com/earth-engine/datasets/catalog/CIESIN_GPWv411_GPW_Population_Density)
    - [NDVI](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_001_VNP13A1)
    - [EVI](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_001_VNP13A1)
## Model
From an overall perspective, the deep learning architecture presented in this article is a prediction model based on the encoder-decoder framework. The encoder is further divided into two parts: the first is the 'Multimodal Encoder', and the second is the 'Backbone Encoder'. In the Multimodal Encoder, we established three blocks to process three types of data, with the aim of increasing their number of channels without changing the height and width of the feature layers. The 'State Block' and 'Dynamic Block's spatiotemporal encoders respectively handle state data with a channel=1 and sequence length set to 'i', and dynamic data with a channel=9 (sequence length refers to the number of the day sequence in the input data). The core component of spatiotemporal encoder selects an architecture widely used for spatiotemporal sequence forecasting, Convolutional Long Short-Term Memory([ConvLSTM](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)) network, for experimentation. The spatial encoder in the 'Constant Block' processes spatial data with a channel size of 9 for constant data, where the architecture chosen is a typical CNN network. Concatenate the outputs of three blocks and feed it into the Backbone Encoder as the final output of the Multimodal Encoder. The Backbone Encoder reduces the resolution of the feature map while increasing the channel depth, effectively compressing spatial information into a more abstract representation. The 'Backbone Decoder' upsamples the bottleneck feature map, restoring it to a size consistent with the input dimensions of 1x64x64. Both Backbone encoder and decoder consist of stacked convolution (transposed convolution for the decoder) and pooling operations.The overall architecture and details of the model are shown in the following figure and table, respectively.
<img width="1043" alt="Figure4_a" src="https://github.com/dazhaxie666/FRP_Prediction/assets/101981022/d134e309-12fc-4c26-b949-4a2091b34529">
| Name             | Layer type  | Input_shape | Output_shape | Kernel size | Kernel numbers | Stride |
|------------------|-------------|-------------|--------------|-------------|----------------|--------|
| State Block      | ConvLSTM(4) | ix1x64x64   | (1x)4x64x64  | (5x5)       | 4              | 1      |
|                  | Conv        | 4x64x64     | 8x64x64      | (3x3)       | 8              | 1      |
|                  | Conv        | 8x64x64     | 16x64x64     | (3x3)       | 16             | 1      |
|                  | Conv        | 16x64x64    | 16x64x64     | (1x1)       | 16             | 1      |
| Dynamic Block    | ConvLSTM(4) | ix1x64x64   | (1x)4x64x64  | (5x5)       | 4              | 1      |
|                  | Conv        | 4x64x64     | 8x64x64      | (3x3)       | 8              | 1      |
|                  | Conv        | 8x64x64     | 16x64x64     | (3x3)       | 16             | 1      |
|                  | Conv        | 16x64x64    | 16x64x64     | (1x1)       | 16             | 1      |
| Constant Block   | Conv        | 6x64x64     | 8x64x64      | (3x3)       | 8              | 1      |
|                  | Conv        | 8x64x64     | 16x64x64     | (3x3)       | 16             | 1      |
|                  | Conv        | 16x64x64    | 16x64x64     | (3x3)       | 16             | 1      |
| Concat           | Concat      | 16x64x64(3) | 48x64x64     | -           | -              | -      |
| Backbone Encoder | Conv        | 16x64x64    | 16x32x32     | (3x3)       | 16             | 2      |
|                  | Maxpooling  | 16x32x32    | 16x16x16     | (2x2)       | -              | 2      |
|                  | Conv        | 32x16x16    | 32x8x8       | (3x3)       | 32             | 2      |
|                  | Maxpooling  | 32x8x8      | 64x4x4       | (2x2)       | -              | 2      |
|                  | Conv        | 64x4x4      | 64x2x2       | (3x3)       | 64             | 2      |
|                  | Maxpooling  | 64x2x2      | 128x1x1      | (2x2)       | -              | 2      |
| Backbone Decoder | TransConv   | 128x1x1     | 64x2x2       | (3x3)       | 64             | 2      |
|                  | TransConv   | 64x2x2      | 32x4x4       | (3x3)       | 32             | 2      |
|                  | TransConv   | 32x4x4      | 16x8x8       | (3x3)       | 16             | 2      |
|                  | Conv        | 16x8x8      | 1x64x64      | (1x1)       | 1              | 1      |
This article uses pytorch for training, and the specific code can be found in this repository.
