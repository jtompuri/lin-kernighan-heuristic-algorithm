# Viikkoraportti 7

## 1. Mitä olen tehnyt tällä viikolla?

Päivitin määrittelydokumentin vastaamaan lopullista toteutusta. Korjasin määrittelydokumentista muutamia kirjoitusvirheitä. Päivitin testausdokumentin vastaamaan viimeisiä muutoksia testeissä ja testikattavuudessa. Päivin myös staattisen testikattavuusraportin kansiossa `htmlconv/`. Poistin versionhallinnasta `.coverage`-tiedoston, mutta säilytin `htmlcov/` kansion, josta löytyy staattinen raportti testikattavuudesta. Poistin `__main__.py` tiedoston docstringistä ylimääräisen ohjeen moduulin ajamisesta.

## 2. Miten ohjelma on edistynyt?

Osa funktioista tai metodeista on pitkiä ja niitä voisi jakaa pienempiin osiin, kuten esimerkiksi `step` ja `alternate_step`. Refaktoroin liian pitkät funktiot käyttämään pienempiä apufunktioita. Pidin huolta, että koodin suorituskyky ei kärsinyt refaktoroinnista.  

## 3. Mitä opin tällä viikolla / tänään?

Koodin selkeyteen vaikuttaa negatiivisesti erityisesti se, että LK-algoritmi on kohtalaisen monimutkainen, kun se kokeilee lukuisia eri heuristiikkoja lyhyemmän polun löytämiseksi. Tämän vuoksi toteutin myös yksinkertaisemman Simple TSP Solver -algoritmin, joka käyttää lähinnä rekursiivisia `2-opt`-siirtoja. Yksinkertaisemman algoritmin koodista saa paremmin kiinni, mistä myös LK-algoritmissa on pohjimmiltaan kyse.  

## 4. Mikä jäi epäselväksi tai tuottanut vaikeuksia? 

Tällä viikolla ei ole ollut epäselviä asioita.

## 5. Mitä teen seuraavaksi?

Sovellus on minun puolestani valmis loppupalautukseen. 