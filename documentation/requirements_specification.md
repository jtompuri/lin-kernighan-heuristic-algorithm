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
- hyvän tehokkuuden rajana pidetään noin 20 sekunnin suoritusaikaa tavalliselle TSP-ongelmalle 

## 3. Termit ja määritelmät

_Kauppamatkustajan ongelma_ voi viitata kysymykseen, onko verkossa Hamiltonin kierros, tai kysymykseen, mikä on lyhimmän Hamiltonin kierroksen pituus. Tässä työssä sillä tarkoitetaan kysymystä, mikä on lyhimmän Hamiltonin kierroksen likiarvo, sillä heuristiset algoritmit eivät tyypillisesti anna parasta ratkaisua. 

Keskeisiä TSP-ongelmaan ja LK-algoritmiin liittyviä käsitteitä:
- _Solmu_ tai _kaupunki (vertex, city)_: TSP:n piste, joita kierretään.
- _Reitti_ tai _kierros (tour)_: solmujärjestys, joka muodostaa suljetun polun kaikkien pisteiden läpi.
- _Etäisyysmatriisi (distance matrix)_: matriisi, joka sisältää solmujen väliset etäisyydet.
- _Täysin kytketty verkko (fully connected network)_: graafi, jossa jokaisella solmulla on yhteys jokaiseen muuhun.
- _K-opt-vaihto (k-opt)_: TSP-heuristiikassa operaatio, jossa katkaistaan ja yhdistetään uudelleen _k_ kaarta uudella tavalla lyhentääkseen kierrosta. LK käyttää vaihtuvaa _k_:ta.
- _Lin-Kernighan heuristiikka (Lin-Kernighan heuristic)_: on edistynyt TSP-heuristiikka, joka käyttä rekursiivista tai ketjutettua k-opt-rakennetta parantaakseen kierrosta inkrementaalisesti.- _Kandidaattireunat (candidate edges)_: valikoidut reunat, joita tarkastellaan mahdollisina vaihdon kohteina (esim. lähimmät _k_ naapurikaupunkia).
- _Hyöty tai parannus (gain)_: kierroksen pituuden vähennys, joka saadaan tietystä k-vaihdosta.
- _Laillinen vaihto (feasible move)_: vaihto, joka tuottaa kelvollisen, suljetun kierroksen ilman kaksoiskäyntejä tai verkosta irtoavia osia.
- _Osittainen polku (partial path): väliaikainen reittijakso k-opt-vaihdon aikana.
- _Rekursiivinen haku (recursive search)_: algoritmin kyky syventää parannuksia useilla peräkkäisillä k-opt-vaihdoksilla, kunnes ei enää saada lisähyötyä.
- _Ketjutettu Lin-Kernighan (chained Lin-Kernighan): strategia, jossa yhdistetään useita LK-vaihteita satunnaisilla aloituksilla uuden, paremman kierroksen löytämiseksi.

## 4. Syötteet ja tulosteet

## 6. Algoritmin kuvaus

## 7. Aika- ja tilavaativuus

## 8. Testausstrategia

## 9. Rajapinnat ja integraatio

## 10. Suoritusympäristö ja vaatimukset

## 11. Liitteet
