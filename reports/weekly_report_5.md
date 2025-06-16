# Viikkoraportti 5

## 1. Mitä olen tehnyt tällä viikolla?

Tein tällä viikolla lähinnä vertaisarvioinnin ja tutustuin vertaisarvioinnissa saamaani palautteeseen.  

## 2. Miten ohjelma on edistynyt?

Sovellus ei ole edistynyt tällä viikolla, sillä olin lomamatkalla. 

En raportoinut edeltäneellä viikolla tekemiäni kokeiluja liittyen sovelluksen suorituskykyyn. Kokeilin nimittäin vaihtaa `Tour`-luokan tietorakenteeksi TSP-kirjan suosittelemaa kahteen suuntaan linkitettyä listaa (_double linked list_) sekä nopeuttaa keskeisten funktioiden ja metodien usein toistuvia silmukoita kääntämällä ne konekielisiksi numba-kirjaston avulla. 

### Kahteen suuntaan linkitetty lista

Tehokkaasti toteutettu kahteen suuntaan linkitetty lista on yksinkertainen ja erittäin nopea tietorakenne LK-algoritmille. Toteutin sen kokeeksi Python luokkana TSP-kirjan mallitoteutuksen mukaisesti. Valitettavasti algoritmin suorituskyky ei parantunut vaan heikentyi. Tämä johtuu luultavasti kahdesta syystä. Ensinnäkin nyt käytössä oleva Numpyn `ndarray` on tehokas ja hyvin optimoitu tietorakenne. Toiseksi kahteen suuntaan linkitetyn listan toteutus Python luokkana ei ole erityisen tehokas tai optimoitu suorituskyvyltään tai muistinhallinnaltaan, kun sitä verrataan esimerkiksi Javan linkitettyihin listoihin. Numpy tallentaa solmut matalan tason C-kielen tietotyyppeinä yhtenevään muistialueeseen, jolloin muistihauissa voidaan hyödyntää tehokkaasti välimuistia. Python luokassa taas jokainen solmu on Python olio, jotka tallennettaan muistiin sattunaisiin osoitteisiin, mikä johtaa useammin välimuistihakujen huteihin.

## Keskeisten silmukoiden kääntäminen konekielelle

Kokeilin käyttää numba-kirjastoa toistuvien silmukoiden nopeuttamiseen. Aluksi tulokset olivat lupaavia, sillä saavutin noin 25 % nopeamman suoritusajan kahden metodin silmukoiden kääntämisellä konekielelle. Valitettavasti tämän jälkeen optimointi ei enää tuottanut merkittää parannusta ja joissain funktioissa jopa hidasti suoritusta. Numban käyttäminen asettaa rajoituksia käännettävälle koodille ja tekee koodista vaikeammin luettavaa, sillä käännettäviin funktioihin lisätään ikään kuin ylimääräinen abstraktiotaso käyttämällä dekoraattoreita. Koin saavutetun hyödyn niin vähäiseksi (noin 25 %), että poistin numban käytöstä. Katsoin, että koodin selkeys ja luettavuus on tärkeämpää kuin marginaalinen parannus suorituskyvyssä. Odotin numballa saavutettavan vähintään yhtä kertaluokkaa vastaava nopeutus, mikä ei toteutunut. Suurin osa TSPLIB95-kirjaston ongelmista ratkeaa hyvin nopeasti ja kahden vaikean ongelman kohdalla pitäisi parantaa algoritmin heuristiikkoja; numba-kirjasto ei auta merkittävästi niiden ratkaisemisessa.

## 3. Mitä opin tällä viikolla / tänään?

Tutustuin saamaani vertaispalautteeseen ja annoin vertaispalautetta toisesta harjoitustyöstä. Opin vertaispalautetta antaessa muun muassa trie- ja n-gram-tietorakenteista sekä graafisen käyttöliittymän toteuttamisesta Pythonilla.   

## 4. Mikä jäi epäselväksi tai tuottanut vaikeuksia? 

Tällä viikolla ei ole ollut epäselviä asioita.

## 5. Mitä teen seuraavaksi?

Seuraavia työvaiheita:
- Jatkan työtä iteraatioiden laskennan parissa
