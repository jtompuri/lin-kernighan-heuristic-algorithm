# Viikkoraportti 1

1. Mitä olen tehnyt tällä viikolla?

Käytin aikaa aiheen valintaan Eigenface-kasvojentunnistuksen ja Lin-Kernighan algoritmin välillä.

Yritin toteuttaa Eigenfaces-algoritmin. Skaalasin ja rajasin Yalefaces-kirjaston 16 380 kuvaa Matlabissa. laskin eigenface-matriisin. Tein matriisille pääkomponenttianalyysin. Laskin 95% tarkkuuteen riittävän määrän pääkomponentteja, joka oli reilu 200 dimensiota. Siirsin aineiston Pythoniin ja toteutin kasvontunnistusalgoritmin, jolla yritin tunnistaa, onko kuvassa kasvoja. Tämä ei toiminut, sillä algoritmi tunnisti tasaisissa pinnoissa kasvoja, mikä johtui luultavasti siitä, että eigenfaces toimii pikemminkin henkilön kasvojen tunnistamisessa muista henkilöistä kuin kasvojen tunnistamiseen kuvasta. Tutkin asiaa ja kasvojentunnistukseen kuvasta on olemassa paljon parempia algoritmeja, joten hylkäsin aiheen.

Valitsin aiheeksi Lin-Kernighan heuristisen algoritmin. Luin alkuperäisen artikkelin vuodelta 1973 ja lainasin ja kuin kirjan _The Traveling Salesman Problem: Computational Study_ luvut TSP:n taustasta ja Lin-Kernighan algoritmista sekä Concorde TSP Solver -ohjelmasta, joka tekijöitä kirjan kirjoittajat ovat. Latasin verkosta ja käänsin Concorden C-kielisen lähdekoodin ja testasin sitä luomiini testiaineistoihin ja TSPLIB-aineistoihin. Latasin ja käänsin myös LKH TSP Solver ohjelman, joka on tällä hetkellä tehokkaimpia TSP-ratkaisija. Tutustuin kursorisesti C-kieliseen lähdekoodiin. 

Pohdin aihetta ja kävin keskustelun Hannu Kärnän kanssa, jonka pohjalta päätin työn rajauksen. Dokumentoin suunnitelman projektin Github-sivulle.    

2. Miten ohjelma on edistynyt?

LK-algoritmi perustuu sitä edeltäneeseen opt2-algoritmiin, joten olen tehnyt kokeiluja yksinkertaisemalla opt2-algoritmilla, joka toimii yllättävän hyvin pienillä optimoinneilla. Ilman optimointeja sain ratkaistua noin 50 solmun verkon ja ottamalla käyttöön NumPyn taulukon tietorakenteeksi ja nopeuttamalla silmukoita kääntämällä ne konekielisiksi numban JIT-käännöstyökalulla, sain ratkaistua 300 solmun verkon. Tätä suuremmilla verkoilla algoritmi luultavasti jäi jumiin paikalliseen minimiin ja Python ikuiseen luuppiin. Tarkoitus on toteuttaa LK-algoritmin "variable k-opt"-algoritmin ja vältettyä paikalliset minimit esimerkiksi "double bridge kick"-algoritmilla. 

3. Mitä opin tällä viikolla / tänään?

Perusideat LK-algoritmissa ja TSP:n heuristisessa ratkaisemisessa. Olen kerrannut kerrattua TSP-algoritmien aikavaativuuksia. Algoritmit opt-2, opt-3 ja variable k-opt. 

4. Mikä jäi epäselväksi tai tuottanut vaikeuksia? Vastaa tähän kohtaan rehellisesti, koska saat tarvittaessa apua tämän kohdan perusteella.

Alkuperäinen artikkeli on vaikeaselkoinen. Onneksi TSP-kirja on huomattavasti selkeämpi ja myös nostaa esiin alkuperäisen artikkelin algoritmin puutteita. Minulle on syntynyt yleiskuva variable k-opt -algoritmista, mutta lähteiden pseudokoodi jättää toteutukseen tosi paljon kysymyksiä esimerkiksi valittavista tietorakenteista. Kirjassa kyllä puhutaan näistä, kuten linkitetyistä listoista ja kaksitasoisista linkitetyistä listoista, mutta toteutuksessa pitää huomioida Pythonin asettamat rajoitukset. Kirja puhuu myös monista LK-algoritmiin tehdyistä parannuksista Concorde-sovellusta kehitettäessä. Jotkut näistä parannuksista voisi olla toteutettavissa tässä projektissa, mutta osa menee selvästi yli realistisesta työn rajauksesta, kuten esimerkiksi "QSopt linear programming solver"-kirjaston hyödyntäminen algoritmissa.

6. Mitä teen seuraavaksi?

Työ jakautuu karkeasti näihin vaiheisiin:
- LK-algoritmin tutkiminen kirjallisuudessa ja ohjelmointi
- testauksen suunnittelu ja toteuttaminen
- Brute force ja opt-2 algoritmien toteutus vertailua varten

## Satunnaisesta 100 solmun polusta
![Satunnainen 100 solmun polku](/kuvat/random_tour.png)

## LK:n ratkaisu 100 solmun polulle (ei optimaalinen)
![LK:n ratkaisema 100 solmun polku](/kuvat/random_tour.png)

## Animaatio 20 solmun ratkaisusta
![Animaatio 20 solmun ratkaisusta](/kuvat/lk_tsp.gif)

