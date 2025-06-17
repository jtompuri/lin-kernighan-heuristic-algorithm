# Viikkoraportti 6

## 1. Mitä olen tehnyt tällä viikolla?

Refaktoroin pääalgoritmin koodin useampaan tiedostoon. Varsinainen LK-algoritmin 
koodi on tiedostossa `lk_algorithm.py`, tiedostoihin liittyvät funktiot tiedostossa 
`tsp_io.py`, apufunktiot tiedostossa `utils.py`, asetukset tiedostossa `config.py`ja 
`main`-funktio tiedostossa `main.py`. Refaktorointi edellytti myös testien refaktoroinnin, 
mikä oli suhteellisen iso työ. 

Refaktoroinnin lisäksi kävin läpi yksityiskohtaisesti testikattavuusraportin ja lisäsin
puuttuvia testejä saavuttaen 100 %:n testikattavuuden. Viimeisten yksittäisten rivien 
kattaminen osoittautui erittäin työlääksi, sillä ohjelman harvinaiseen haaraan päätyminen 
edellytti juuri tietynlaista TSP-ongelmaa. Yhden puuttuvan rivin ohitin, sillä koodi 
ei koskaan päädy tälle riville. Rivi on olemassa puolustavan ohjelmointistrategian 
vuoksi. Toisin sanoen riville päädytään vain siinä tapauksessa, että muualla koodissa 
tehdään ohjelmointivirhe.


Kävin läpi käytetyt kirjastot ja poistin tarpeettomat, kuten esimerkiksi `tsplib95`
-kirjaston, jota ei ole enää ylläpidetty ja jota käytin vain TSP-ongelmien luonnissa. 
Korvasin kirjastosta käytetyt piirteet omalla koodilla. Jaoin asennettavat kirjastot
ajonaikaisiin kirjastoihin (`requirements.txt`) ja kehitykseen tarvittaviin kirjastoihin 
(`requirements-dev.txt`). Päivitin asennusohjeet tämän jaon mukaisesti.

Tein sovelluksesta ajettavan Python moduulin ja muutin README:n ohjeen vastaamaan tätä. 

## 2. Miten ohjelma on edistynyt?

Sovellus on nyt rakenteeltaan modulaarisempi. Täydellinen testikattavuus on saavutettu. 

## 3. Mitä opin tällä viikolla / tänään?

Opin lisää testikattavuuden parantamisesta, kirjastojen hallinnasta ja sovelluksen 
ajonaikaisesta ympäristöstä ja kehitysympäristöstä.

## 4. Mikä jäi epäselväksi tai tuottanut vaikeuksia? 

Tällä viikolla ei ole ollut epäselviä asioita.

## 5. Mitä teen seuraavaksi?

Tutkin vielä iteraatioiden lisäämistä tulostukseen.