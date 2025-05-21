# Määrittelydokumentti

## 1. Johdanto

Harjoitustyössä toteutetaan Lin-Kernighan heuristinen algoritmi, joka antaa likimääräisen ratkaisun symmetriseen kauppamatkustajan ongelmaan (_Traveling Salesman Problem_, TSP). Kauppamatkustajan ongelmaa pidetään TSP-kovana ja tarkan ratkaisun antava kaikki reitit läpikäyvän algoritmin aikavaativuutena pidetään O(n!), mikä tekee algoritmista käyttökelvottoman jo 20 kaupungin verkoissa. Tämän vuoksi TSP-ongelmaa on yritetty ratkaista heuristisilla algoritmeilla, joista Lin-Kernighan heuristinen algoritmi (LK) on osoittautunut yhdeksi tehokkaimmista. TSP-ongelman tutkimuksessa kehitettyjä heuristisia algoritmeja käytetään laajasti erilaisissa optimointia vaativissa tehtävissä kuten logistiikassa, teollisuudessa, biotieteissä, autonomisissa järjestelmissä, tietokonepeleissä ja verkkopalveluissa.

## 2. Rajaus ja soveltamisala

LK-algoritmin toteutuksen rajauksia, jotta toteutuksen laajuus pysyy hallittavana:
- käsitellään vain symmetrisiä TSP-ongelmia eli ei-suunnattuja, täysin kytketty verkkoja
- käsitellään vain TSP-ongelmia, joissa on euklidinen, kaksiuloitteinen geometria
- käytetään mallina _The Traveling Salesman Problem: A Computational Study_ kuvausta LK-algoritmista
- toteutuksesta jätetään pois piirteet, joita ei voi toteuttaa tehokkaasti Pythonilla 
- toteutuksessa tavoitellaan hyvää tasapainoa ratkaisun laadun ja tehokkuuden välillä
- hyvän laadun rajana pidetään noin 5 %:n ylitystä suhteessa optimaaliseen ratkaisuun
- hyvän tehokkuuden rajana pidetään noin 20 sekunnin suoritusaikaa tavalliselle TSP-ongelmalle. 

## 3. Termit ja määritelmät

_Kauppamatkustajan ongelma_ voi viitata kysymykseen, onko verkossa Hamiltonin kierros, tai kysymykseen, mikä on lyhimmän Hamiltonin kierroksen pituus. Tässä työssä sillä tarkoitetaan kysymystä, mikä on lyhimmän Hamiltonin kierroksen likiarvo, sillä heuristiset algoritmit eivät tyypillisesti anna parasta ratkaisua. 

Keskeisiä TSP-ongelmaan ja LK-algoritmiin liittyviä käsitteitä:
- _Solmu_ tai _kaupunki (vertex, city)_: TSP:n piste, joita kierretään.
- _Reitti_ tai _kierros (tour)_: solmujärjestys, joka muodostaa suljetun polun kaikkien pisteiden läpi.
- _Etäisyysmatriisi (distance matrix)_: matriisi, joka sisältää solmujen väliset etäisyydet.
- _Täysin kytketty verkko (fully connected network)_: graafi, jossa jokaisella solmulla on yhteys jokaiseen muuhun.
- _K-opt-vaihto (k-opt)_: TSP-heuristiikassa operaatio, jossa katkaistaan ja yhdistetään uudelleen _k_ kaarta uudella tavalla lyhentääkseen kierrosta. LK käyttää vaihtuvaa _k_:ta.
- _Lin-Kernighan heuristiikka (Lin-Kernighan heuristic)_: edistynyt TSP-heuristiikka, joka käyttää rekursiivista tai ketjutettua k-opt-rakennetta parantaakseen kierrosta inkrementaalisesti.
- _Kandidaattireunat (candidate edges)_: valikoidut reunat, joita tarkastellaan mahdollisina vaihdon kohteina (esim. lähimmät _k_ naapurikaupunkia).
- _Hyöty tai parannus (gain)_: kierroksen pituuden vähennys, joka saadaan tietystä k-vaihdosta.
- _Laillinen vaihto (feasible move)_: vaihto, joka tuottaa kelvollisen, suljetun kierroksen ilman kaksoiskäyntejä tai verkosta irtoavia osia.
- _Osittainen polku (partial path)_: väliaikainen reittijakso k-opt-vaihdon aikana.
- _Rekursiivinen haku (recursive search)_: algoritmin kyky syventää parannuksia useilla peräkkäisillä k-opt-vaihdoksilla, kunnes ei enää saada lisähyötyä.
- _Ketjutettu Lin-Kernighan (chained Lin-Kernighan)_: strategia, jossa yhdistetään useita LK-vaiheita satunnaisilla aloituksilla uuden, lyhyemmän kierroksen löytämiseksi.

## 4. Syötteet ja tulosteet

LK-algoritmin syötteenä käytetään tutkimuskirjallisuudessa standardia TSPLIB-tiedostomuotoa (.tsp) ja TSPLIB95-kirjaston TSP-ongelmia, jotka ovat symmetrisiä ja joissa on euklidinen,  kaksiuloitteinen geometria ja joihin löytyy optimaalinen ratkaisu erillisenä tiedostona (.opt.tour). Satunnaisia _n_ solmun verkkoja voi halutessaan luoda _create_tsp_files.py_ ohjelmalla, mutta koska verkkojen optimaalista ratkaisua ei yleensä tunneta, niin satunnaisia verkkoja ei voi käyttää algoritmin laadun arviointiin, mutta niilä voi tutkia algoritmin suorituskykyä. 

Algoritmi tulostaa terminaaliin algoritmin konfiguraation tiedot ja yhteenvedon TSP-ongelmista, joka sisältää ongelman tunnisteen, kierroksen optimaalisen ratkaisun pituuden, heuristisen kierroksen pituuden, poikkeaman prosentteina (Gap) ja suoritusajan. Lisäksi esitetään tulosten summa- tai keskilukuja. Algoritmi piirtää kuvaajat kaikista TSP-ongelmista. Kuvaajissa esitetään heuristinen kierros kiinteänä viivana ja optimaalinen kierros pisteviivana eri värillä sekä poikkeama. Kuvaajista näkee kuinka lähellä heuristinen ratkaisu on optimaalista ratkaisua. 

## 6. Algoritmin kuvaus

LK-algoritmin toteutuksessa noudatetaan kirjan _The Traveling Salesman Problem: A Computational Study_ esitystä Lin-Kernighan heuristiikasta. Esitystä voi pitää kanonisena, sillä kirjan kirjoittajat ovat keskeisiä alan tutkijoita ja _Concorde TSP Solver_ sovelluksen kehittäjiä. Kirjassa käytetään esimerkkinä _Concorde_ sekä _LKH TSP Solveria_, jotka ovat tällä hetkellä tehokkaimpia TSP-ongelman ratkaisuun kehitettyjä sovelluksia.

### 6.1 Vertailualgoritmit

Toteutan vertailua varten algoritmin (_exact TSP solver_), joka käy läpi annetun verkon kaikki reittiyhdistelmät. Tämä algoritmi tuottaa aina tarkan ratkaisun, mutta sen aika vaativuus on _O(n!)_, sillä reittikombinaatioiden määrä ei-suunnatussa, täysin kytketyssä verkossa saadaan kaavalla (n - 1)!/2. Käytännössä tavallisella tietokoneella voi ratkaista 10 solmun verkon, mutta jo 12 solmun verkko on usein jo liian vaativa etenkin Pythonin kaltaiselle kielelle.  

Toinen vertailualgoritmi on LK-algoritmin erityistapaus 2-opt-algoritmi. LK-algoritmin idean taustalla oli edeltänyt 2-opt-vaihtojen tutkimus TSP-ongelmien ratkaisussa. Voidaan ajatella, että kun poistetaan LK-algoritmista kaikki muut heuristiikat, niin LK-algoritmista tulee 2-opt algoritmi. Koska 2-opt-algoritmi on merkittävästi yksinkertaisempi, niin se toimii hyvänä johdantona varsinaisen LK-algoritmin toteutukselle. Yksinkertaisuus johtaa toisaalta siihen, että hyvin optimoitu 2-opt-algoritmi on erittäin nopea suorituskyvyltään, toisaalta se tuo esiin heurististen algoritmien keskeisen ongelman eli jumittumisen paikalliseen minimiin. 

### 6.2 LK-algoritmin mallitoteutus

_The Traveling Salesman Problem_-kirjan luku 15 _Tour Finding_ esittelee LK-algoritmin tavoitteet ja mallitoteutuksen. Mallitoteutuksessa pyritään lähes optimaaliseen ratkaisuun järkevässä suoritusajassa, mikä on myös tämän harjoitustyön tavoitteena. LK-algoritmista on toisilla kirjoittajilla, kuten esimerkiksi nopeuden korostaminen ja tyytyminen vähemmän optimaalisiin ratkaisuihin. 

![Kahden kaaren vaihto](/images/figure_15_1.png)

Algoritmin idea perustuu polun kaarien vaihtoihin, joilla käännetään reitin osan solmujen järjestys päinvastaiseksi. Jos uuden reitin pituus on lyhyempi kuin alkuperäisen, säilytetään vaihto. Muuten vaihto hylätään. Yksinkertaisin vaihto on kahden kaaren vaihto, mutta yhtä hyvin voidaan vaihtaa kolme tai useampi kaari keskenään. Lisäksi kaaren vaihtoja voidaan tehdä peräkkäin useita ja hyväksyä, että reitin välivaiheen pituus on suurempi kuin alkuperäinen, jolloin on mahdollista löytää usean vaihdon tuloksena alkuperäistä lyhyempi reitti.  

LK-algoritmin suunnittelussa on panostettu erityisesti kaarien vaihtojen optimointiin. Haasteena kaarien vaihdossa on toisaalta se, että pitäisi välttää turhia kaarenvaihtoja, ja toisaalta etsiä rohkeasti lyhyempää reittiä eikä tyytyä paikallisesti optimaaliseen ratkaisuun. Käytännösssä LK-algoritmi toki lähes poikkeuksetta päätyy paikalliseen eikä yleiseen optimaaliseen ratkaisuun pois lukien hyvin yksinkertaiset verkot. Rekursiivisen haun leveyttä, eli kuinka monta naapurisolmua otetaan mukaan hakuun, rajoitetaan suhteessa haun syvyyteen. Mallitoteutuksessa syvyydelle ei ole asetettu ylärajaa, mutta käytännön toteutuksissa käytetään usein 10-12 ylärajaa rekursioille. Tyypillisesti haun leveys on rajoitettu ensimäisellä tasolla 5 naapuriin, toisella tasolla 5 naapuriin ja tätä seuraavilla tasoilla yhteen naapuriin.

![Algoritmi step()-funktiolle](/images/algorithm_15_1.png)




## 7. Aika- ja tilavaativuus

## 8. Testausstrategia

## 9. Rajapinnat ja integraatio

## 10. Suoritusympäristö ja vaatimukset

## 11. Liitteet
