---
title: "AI4EO-An Opportunity for Computation Sustainability"
description: This post introduces how Earth Observation (EO)  and Artificial Intelligence (AI) could be used for Computational Sustainability.
toc: true
comments: true
layout: post
categories: [Machine learning, Deep learning, Generative models, EO]
image: images/post/eo.jpg
author: Anthony Faustine & Shridhar Kulkarni
---

## Introduction

Computational Sustainability focuses on developing computational models, methods, and tools to help policymakers design more effective solutions and policies for sustainable development. The advancement of Information and Communication Technologies (ICT), particularly Earth Observation (EO) and Artificial intelligence (AI) offer prospects of addressing sustainability challenges.A more in-depth explanation about the above project can be viewed in this video:

{% include youtube.html content="https://www.youtube.com/watch?v=vDC5T9Wvgeo" %}
Earth observations (EO) are data and information about the planet’s physical, chemical, and biological systems.  It involves the collection, analysis, and presentation about the status of, and changes in, the natural and human-made environment. The most common sources of EO data include drones,  land stations, and satellites. While drones capture high-resolution images on a small scale, satellites generate growing amounts of multi-resolution and multi-bands imagery and other data sources for the whole Earth. These data could be used to create all kinds of different products so that scientists, policymakers, and even everyday citizens can understand the past, present, and future trends in the Earth systems. Figure [below](https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/raster-bands.htm) shows multiband imagery from satellites by the electromagnetic 

![](images/fastpages_posts/actions/Multispectral_bands.png)

On the other hand, AI is an area of computer science devoted to developing systems that can learn (from data) to make decisions and predictions within specific contexts. Indeed, AI technology can extract more in-depth insights from datasets than other techniques. Over recently, AI has been used with success in solving complex problems in several domains such as machine translation, computer vision, autonomous cars, to mention a few. Machine learning and specifically, computer vision models provide explicitly useful and practical approaches for analyzing and extracting relevant information from EO imagery data. Recently deep learning models and specifically Convolution Neural Networks (CNNs) have proven effective in several computer vision tasks such as object detection, classification, and video processing, image generations, and image captioning, to mention a few. These models could be applied to detect and classify objects from complex EO imagery at a larger scale. Figure 2 presents AI capability for object detection and using computer vision techniques for multiband satellite images. This image has been taken from [here](https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/raster-bands.htm)).

![](images/fastpages_posts/actions/EO_AI.png)

Applying these techniques to EO data will make it easy to efficiently automate the recognition of known and unknown patterns at large-scale. This is likely to reveal useful insights and opportunities for addressing sustainability challenges. For example, AI models could be applied to perform automated change detection, crop mapping, and yield estimation from high-resolution imagery in a larger-scale. The fusion of EO data and other data sources such as geo-referenced, demographics, and social-network data can be used to facilitate the more targeted developmental challenge. For instance, it has been demonstrated that the AI model can be used to predict the poverty level by analyzing satellite imagery, night lights, and demographic data.  Figure below shows the different application domains of EO.Image has been taken from [here](https://www.sentinel-hub.com/) and [here](https://www.agrilinks.org/post/enhancing-earth-observation-solutions-agriculture-machine-learning)

![](images/fastpages_posts/actions/Applications_2.png)             


##  EO data sources

There are many EO data sources made available recently. These data sources offer virtual visualization of any location on earth with resolution ranging from 5 centimeters to 120 meters depending on the instruments of satellites, airbus, or drones. The data sources are published as public or commercial data sources.

**Public EO data providers**

The EO puplic data providers are public service framework that allows full, free and open access to all data collected. [Copernicus](https://www.copernicus.eu/en/about-copernicus) and [Landsat](https://landsat.gsfc.nasa.gov/about/) are the famous and largest public satellite data providers. 
Landsat s one of the world's largest satellite image providers. It is a joint program of the National Aeronautics and Space Administration (NASA) and the United States Geological Survey (USGS). It provides access to satellites of the Landsat family, which have access over the archival of 50 years of earth data. Landsat satellites collect data on the forests, farms, urban areas, and water sources, generating the longest continuous record. The freely available information is used to understand environmental change better, [manage agricultural practices](https://landsat.gsfc.nasa.gov/how_landsat_helps/agriculture/), [allocate scarce water resources](https://landsat.gsfc.nasa.gov/how_landsat_helps/water), [monitor the extent and health of forests](https://landsat.gsfc.nasa.gov/how_landsat_helps/forest-management/) and respond to natural disasters, and more. Data can be accessed using LandsLook Viewer, USGS GloVis, Earth Explorer, Free Web-Enabled Landsat Data (WELD). More information is available [here](https://landsat.gsfc.nasa.gov/data/where-to-get-data/).

Copernicus is managed by the Europe Unions EO program and collect data from a constellation of 6 families of satellites, known as Sentinels. Each Sentinel mission focuses on different but interrelated aspects of EO, including [Atmospheric monitoring (Sentinels 4 and 5)](http://atmosphere.copernicus.eu), [Marine environment monitoring (Sentinel-3)](http://marine.copernicus.eu), [Land monitoring (Sentinel-2)](http://land.copernicus.eu), [Climate Change](http://climate.copernicus.eu) and [Emergency management](http://emergency.copernicus.eu). Currently Copernicus produces 12 terabytes per day of data for the 6 families of satellites, known as "Sentinels." The data are  open access and can be freely downloaded using [Copernicus Open Access Hub]. 

**Commercial data providers**

The commercial satellite imagery providers provide access to data with high resolution with 3 centimeters to 10 meters. These services are paid and have good archival imagery. The most popular commercial EO imagery providers include; [Planet Labs](https://www.planet.com/), [DigitalGlobe](https://www.digitalglobe.com/) and [Airbus](https://www.intelligence-airbusds.com/en/8692-pleiades). 

[Planet Labs](https://www.planet.com/) provides access to a wide range of satellite data. It provides access to SkySAT families and RapidEye satellites. With 120+ satellites in orbit, Planet can image anywhere on Earth’s landmass daily, at 3 - 5-meter resolution. Planet processes and delivers imagery quickly and efficiently. Planet’s platform downloads and processes 11+ TB of data daily, enabling customers to build and run analytics at scale. Users can access Planet's data, using the paid planet API. Nevertheless, university researchers, academics, and scientists apply for free access as decribed in this [link](https://www.planet.com/markets/education-and-research/).

The [DigitalGlobe](https://www.digitalglobe.com/) is similar to Planet Labs and provides data access to a full range constellation of satellites in orbit. It provides access to EarlyBird-1, IKONOS, QuickBird, GeoEye-1, a family of WorldView satellites. It offers a high resolution of up to 30cm, showing crisp details, satellite imagery, geospatial information, and location-based intelligence. Recently, DigitalGlobe has started providing 0.4m resolution imagery today, which is one of the best in the business. 

On the other hand, the Airbus, with Pleiades and SPOT missions, provide very high-resolution multispectral twin satellites with 0.5 meters and 1.5-meter resolution, respectively. These imagery data are particularly suitable for emergency response and up-to daily change detection.

![](images/fastpages_posts/actions/Satellites_commrcial_coopernicus.png) 


### AI ready EO datasets
Building ML applications for EO requires access to both EO data and their ground truth. Creating such a data-set is time-consuming and costly. As a result different organisations provide ready-to-use EO dataset which allow ML and GIS researchers and other stakeholders to build and test their ML application specific to EO. Radiant MLHub and Spacenet are the two notable EO training data providers. [Radiant MLHub](www.mlhub.earth) is an open library for geospatial training data to advance machine learning applications on EO. It hosts open training datasets generated by Radiant Earth Foundation's team as well as other training data catalogs contributed by Radiant Earth's partners. The data provided by Radiant MLHub are stored using a SpatioTemporal Asset Catalog (STAC) compliant catalog and exposed through a standard API. These data are open to anyone to use. It also free stores, register and share your dataset. 

The [Spacenet](https://spacenet.ai/datasets/), on the other hand, provides access to high-quality geospatial data for developers, researchers, and startups with a specific focus on the four open-source key pillars: data, challenges, algorithms, and tools. It also hosts challenges that focus on applying advanced machine learning techniques to solve difficult mapping challenges. The SpaceNet Dataset is hosted as an [Amazon Web Services (AWS) Public Dataset](https://registry.opendata.aws/), which is open for geospatial machine learning research. The dataset consists of well-annotated and very high-resolution satellite imagery with foundational mapping features such as building footprints or road networks.

[Kaggle](https://www.kaggle.com), a world's largest data science community with powerful tools and resources, is another source of EO training datasets which host several machine learning challenges EO imagery. This challenges includes [Dstl Satellite Imagery Feature Detection](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection), [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection) and [Draper Satellite Image Chronology](https://www.kaggle.com/c/draper-satellite-image-chronology) to mention a few.

![](images/fastpages_posts/actions/ai_ready.png) 

             
