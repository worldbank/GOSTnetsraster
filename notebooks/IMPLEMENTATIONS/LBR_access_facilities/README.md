# Liberia ID centre travel time analysis

This analysis evaluates travel time to existing and proposed ID centers in Liberia.
<img src='maps/LBR_existing_and_proposed_service_centers.png' alt='Map of primary roads and service centers' width='600'/>

This was done using friction surface analysis to measure travel time to service centers, and map marketsheds for service centers
The following analysis were undertaken to quantify accessibility to ID centers in Liberia:

| Question | Output file |
|---|---|
| How many people are within 60 mins and 120 mins drive time to existing ID centers, summarized at ADM1 and ADM2 levels? | ADM2_tt_existing_ID.csv |
| How many exclusive users are within 120 mins drive time to existing ID centers (Marketsheds)? | Marketshed_population_existing_ID.csv |
| What is the population outside 120 min drive time of existing centers (unserved population)? | ADM2_tt_existing_ID.csv |
| How many people are within 60, 120 min drive time to proposed centers? | ADM2_tt_proposed_ID.csv |
| What are the Marketsheds for proposed centers? | Marketshed_population_proposed_ID.csv |
| How to prioritize proposed centers sequentially to serve the most population with the fewest centers? | Top_100_proposed_ID_sites.gpkg |

## Maps of results

<img src='maps/LBR_tt_raster_existing_ID.png' alt='Travel time to existing ID centers map' width='600'/>

The friction surface analysis measures travel time to the nearest destination from every location in the country, accounting for road networks and landcover-based travel speeds.

<img src='maps/LBR_tt_vector_existing_ID.png' alt='60 min drive-time existing ID centers' width='600'/>

Drive-time vectors show areas that can reach a service center within a given time threshold. This differs from the above analysis in that             each destination (ID centre) is treated independently, so the drive areas can overlap

<img src='maps/LBR_market_sheds_existing_ID.png' alt='LBR Market Sheds existing ID centers' width='600'/>

Marketsheds show the exclusive service area for each service center, i.e., the area that is closest to a given center compared to all other centers.
<img src='maps/LBR_proposed_top_100_and_existing_service_centers.png' alt='Map of proposed top 100 sites and existing sites' width='600'/>

The proposed top 100 sites are selected sequentially to maximize the unserved population within 60 mins drive time.
