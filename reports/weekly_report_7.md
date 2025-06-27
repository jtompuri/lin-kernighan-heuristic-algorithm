# Viikkoraportti 7

## 1. Mitä olen tehnyt tällä viikolla?

Päivitin määrittelydokumentin vastaamaan lopullista toteutusta. Päivitin testausdokumentin vastaamaan viimeisiä muutoksia testeissä ja testikattavuudessa. Päivin myös staattisen testikattavuusraportin kansiossa `htmlconv/`. 

Poistin versionhallinnasta `.coverage`-tiedoston, mutta säilytin `htmlcov/` kansion, josta löytyy staattinen raportti testikattavuudesta. Poistin `__main__.py` tiedoston docstringistä ylimääräisen ohjeen moduulin ajamisesta.

## 2. Miten ohjelma on edistynyt?

Osa funktioista tai metodeista on pitkiä ja niitä voisi jakaa pienempiin osiin, kuten esimerkiksi `step` ja `alternate_step`. Päätin kuitenkin säilyttää 1:1-vastaavuuden alkuperäiseen TSP-kirjan pseudokoodiin ja siinä käytettyyn koodin rakenteeseen. Toteutuksen rivimäärä on kasvanut suhteessa alkuperäiseen pseudokoodiin, kun olen lisännyt koodiin esimerkiksi pisimmän sallitun suoritusajan (`deadline`-parametri) vaatimat tarkistukset. 

## 3. Mitä opin tällä viikolla / tänään?

Koodin selkeyteen vaikuttaa negatiivisesti erityisesti se, että LK-algoritmi on kohtalaisen monimutkainen, kun se kokeilee lukuisia eri heuristiikkoja lyhyemmän polun löytämiseksi. Tämän vuoksi toteutin myös yksinkertaisemman Simple TSP Solver -algoritmin, joka käyttää lähinnä rekursiivisia `2-opt`-siirtoja. Yksinkertaisemman algoritmin koodista saa paremmin kiinni, mistä myös LK-algoritmissa on pohjimmiltaan kyse.  

## 4. Mikä jäi epäselväksi tai tuottanut vaikeuksia? 

Tällä viikolla ei ole ollut epäselviä asioita.

## 5. Mitä teen seuraavaksi?

Sovellus on minun puolestani valmis loppupalautukseen. 