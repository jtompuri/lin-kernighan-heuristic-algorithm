# Viikkoraportti 8

## 1. Mitä olen tehnyt tällä viikolla?

Olen viimeistellyt harjoitustyötä demotilaisuudessa saamani palautteen ja kysymysten perusteella. Demotilaisuudessa ohjaaja esitti kysymyksen kierroksen alkujärjestyksestä ja sen vaikutuksesta tuloksiin. Olen aiemmin kokeillut `nearest neighbor`-algoritmia alkujärjestyksen luomisessa, mikä ei vaikuttanut testiaineistolla merkittävästi tuloksiin. Nyt heräsi kuitenkin kysymys koskien suuria verkkoja, joissa voisi ajatella alkujärjestyksestä olevan hyötyä. Kokeilin 10 000 solmun verkolla tätä ja huomasin, että algoritmin oletusjärjestys, jossa solmut on niiden luonnollisessa järjestyksessä, on merkittävästi huonompi kuin `nearest neighbor`-algoritmilla toteutettu alkujärjestys, joten päätin toteuttaa vaihtoehtoisia algoritmeja alkujärjestyksen luomiseen. Otin mallia Concorden mukana tulevan `linkern`-ratkaisijan alkujärjestyksistä ja totetin algoritmit: `nearest neighbor`, `greedy`, `boruvka`ja `qboruvka`. Alkujärjestys auttaa merkittävästi vaikeiden tsp-ongelmien, kuten `pcb442.tsp` ja `pr1002.tsp` ratkaisussa. 

Lisäsin sovellukseen usean ytimen tuen, kun tehdään usean tsp-ongelman eräajoa. Tällöin yksi ydin ratkaisee aina yhtä tsp-ongelmaa. Tämä nopeuttaa merkittävästi suorituskykyä. Stokastista syistä tulokset rinnakkaisessa ja peräkkäisessä prosessoinnissa eroavat jonkin verran toisistaan. Tämä näkyy niin, että eri ongelmat osoittautuvat helpoiksi tai vaikeiksi riippuen prosessointi tavasta. 

Toteutin sivuprojektina [erilliseen repoon](https://github.com/jtompuri/weighted-voronoi-stippling) kurssille alunperin suunnittelemani taideprojektin, jossa tuotetaan valokuvasta pistevarjostustekniikalla tsp-tiedosto, joka tämän jälkeen se ratkaistaan ulkoisella tsp-ratkaisijalla, jolloin lopputulokseksi saadaan kierros, josta alkuperäinen valokuva on tunnistettavissa. Toteutin LK-algoritmiin kierroksen tallentamisen `.tour`-tiedostoksi tätä varten. Tekniikan on tehnyt tunnetuksi Robert Bosch. Esimerkki tekniikasta on alla. 

Harjoitustyön viimeistely on sisältänyt seuraavia työvaiheita:
- kansiorakenteen selkeyttäminen
- turhien tiedostojen poistaminen
- README-tiedoston tiivistäminen
- komentoriviargumenttien lisääminen
- toteutus- ja testausdokumenttien päivitys. 

## 2. Miten ohjelma on edistynyt?

Sovellus on edistynyt hyvin, vaikka olen käyttänyt paljon aikaa sivuprojektiin.

## 3. Mitä opin tällä viikolla / tänään?

Opittuja asioita:
- usean ytimen tuki Python-sovelluksissa
- algoritmin Weighted Voronoi Stippling toteutus
- numba-kirjaston käyttäminen silmukoiden nopeuttamiseen
- GPU:n käyttäminen silmukoiden nopeuttamiseen (vaatii CUDA-tuen)  

## 4. Mikä jäi epäselväksi tai tuottanut vaikeuksia? 

Tällä viikolla ei ole ollut epäselviä asioita.

## 5. Mitä teen seuraavaksi?

Sovellus on minun puolestani valmis loppupalautukseen. 

## Esimerkki sivuprojektista

Alkuperäinen valokuva:

![Alkuperäinen valokuva](/images/example-1024px.png)

Pistevarjostustekniikalla toteutettu kuva ([weighted-voronoi-stippling](https://github.com/jtompuri/weighted-voronoi-stippling)):

![Pistevarjostustekniikka](/images/stippling_example-1024px_10000.png)

LK-algoritmilla ratkaistu kierros:

![Ratkaistu kierros](/images/example-1024px_10000.png)
