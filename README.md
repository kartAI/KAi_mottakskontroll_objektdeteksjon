# KAi_mottakskontroll_objektdeteksjon
Teste KartAI-algoritmer på utvalgt område i et GeoVekst-prosjekt, for å sjekke om KartAI-algoritmene kan detektere bygninger på et nivå som kan være til hjelp ved mottakskontroll av bygningsdata konstruert fra flybilder.

I dette projectet er det laget to scripts for å generere treningsdata for YOLO objket deteksjoins modell og YOLO instance segmentation modell. 

Scriptene er laget med tanke på FKB-Bygning data av laget takkant og flybilder for området Farsund. Disse må legges til og definere riktige stier. FKB-Bygning er på Geopackage format og Flyfotoen er på tiff format. Det er viktig at all dataen bruker samme koordinatsystem. 
