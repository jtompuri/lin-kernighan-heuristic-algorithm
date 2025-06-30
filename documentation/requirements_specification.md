# Määrittelydokumentti

Janne Tompuri, Matemaattisten tieteiden kandiohjelma ja maisteriohjelma

## 1. Johdanto

Harjoitustyössä toteutetaan Lin–Kernighan-heuristinen algoritmi, joka antaa likimääräisen ratkaisun symmetriseen kauppamatkustajan ongelmaan (_Traveling Salesman Problem_, TSP). Kauppamatkustajan ongelmaa pidetään TSP-kovana ja tarkan ratkaisun antava kaikki reitit läpikäyvän algoritmin aikavaativuutena pidetään $O(n!)$, mikä tekee algoritmista käyttökelvottoman jo 20 kaupungin verkoissa. Tämän vuoksi TSP-ongelmaa on yritetty ratkaista heuristisilla algoritmeilla, joista Lin–Kernighan-heuristinen algoritmi (LK) on osoittautunut yhdeksi tehokkaimmista. TSP-ongelman tutkimuksessa kehitettyjä heuristisia algoritmeja käytetään laajasti erilaisissa optimointia vaativissa tehtävissä kuten logistiikassa, teollisuudessa, biotieteissä, autonomisissa järjestelmissä, tietokonepeleissä ja verkkopalveluissa.

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
- _Lin–Kernighan-heuristiikka (Lin–Kernighan heuristic)_: edistynyt TSP-heuristiikka, joka käyttää rekursiivista tai ketjutettua k-opt-rakennetta parantaakseen kierrosta inkrementaalisesti.
- _Kandidaattireunat (candidate edges)_: valikoidut reunat, joita tarkastellaan mahdollisina vaihdon kohteina (esim. lähimmät _k_ naapurikaupunkia).
- _Hyöty tai parannus (gain)_: kierroksen pituuden vähennys, joka saadaan tietystä k-vaihdosta.
- _Laillinen vaihto (feasible move)_: vaihto, joka tuottaa kelvollisen, suljetun kierroksen ilman kaksoiskäyntejä tai verkosta irtoavia osia.
- _Osittainen polku (partial path)_: väliaikainen reittijakso k-opt-vaihdon aikana.
- _Rekursiivinen haku (recursive search)_: algoritmin kyky syventää parannuksia useilla peräkkäisillä k-opt-vaihdoksilla, kunnes ei enää saada lisähyötyä.
- _Ketjutettu Lin–Kernighan (chained Lin–Kernighan)_: strategia, jossa yhdistetään useita LK-vaiheita satunnaisilla aloituksilla uuden, lyhyemmän kierroksen löytämiseksi.

## 4. Syötteet ja tulosteet

LK-algoritmin syötteenä käytetään tutkimuskirjallisuudessa standardia TSPLIB-tiedostomuotoa (.tsp) ja TSPLIB95-kirjaston TSP-ongelmia, jotka ovat symmetrisiä ja joissa on euklidinen,  kaksiuloitteinen geometria ja joihin löytyy optimaalinen ratkaisu erillisenä tiedostona (.opt.tour). Satunnaisia _n_ solmun verkkoja voi halutessaan luoda `create_tsp_problem.py` ohjelmalla, mutta koska verkkojen optimaalista ratkaisua ei yleensä tunneta, niin satunnaisia verkkoja ei voi käyttää algoritmin laadun arviointiin, mutta niilä voi tutkia algoritmin suorituskykyä. 

Algoritmi tulostaa terminaaliin algoritmin konfiguraation tiedot ja yhteenvedon TSP-ongelmista, joka sisältää ongelman tunnisteen, kierroksen optimaalisen ratkaisun pituuden, heuristisen kierroksen pituuden, poikkeaman prosentteina (Gap) ja suoritusajan. Lisäksi esitetään tulosten summa- tai keskilukuja. Algoritmi piirtää kuvaajat kaikista TSP-ongelmista. Kuvaajissa esitetään heuristinen kierros kiinteänä viivana ja optimaalinen kierros pisteviivana eri värillä sekä poikkeama. Kuvaajista näkee kuinka lähellä heuristinen ratkaisu on optimaalista ratkaisua. 

## 6. Algoritmin kuvaus

LK-algoritmin toteutuksessa noudatetaan kirjan _The Traveling Salesman Problem: A Computational Study_ esitystä Lin–Kernighan-heuristiikasta. Esitystä voi pitää kanonisena, sillä kirjan kirjoittajat ovat keskeisiä alan tutkijoita ja _Concorde TSP Solver_ sovelluksen kehittäjiä. Kirjassa käytetään esimerkkinä _Concorde_ sekä _LKH TSP Solveria_, jotka ovat tällä hetkellä tehokkaimpia TSP-ongelman ratkaisuun kehitettyjä sovelluksia.

### 6.1 Vertailualgoritmit

Toteutan vertailua varten algoritmin (_Exact TSP Solver_), joka käy läpi annetun verkon kaikki reittiyhdistelmät. Tämä algoritmi tuottaa aina tarkan ratkaisun, mutta sen aika vaativuus on $O(n!)$, sillä reittikombinaatioiden määrä ei-suunnatussa, täysin kytketyssä verkossa saadaan kaavalla (n - 1)!/2. Käytännössä tavallisella tietokoneella voi ratkaista 10 solmun verkon, mutta jo 12 solmun verkko on usein jo liian vaativa etenkin Pythonin kaltaiselle kielelle.  

Toinen vertailualgoritmi on LK-algoritmin erityistapaus 2-opt-algoritmi. LK-algoritmin idean taustalla oli edeltänyt 2-opt-vaihtojen tutkimus TSP-ongelmien ratkaisussa. Voidaan ajatella, että kun poistetaan LK-algoritmista kaikki muut heuristiikat, niin LK-algoritmista tulee 2-opt algoritmi. Koska 2-opt-algoritmi on merkittävästi yksinkertaisempi, niin se toimii hyvänä johdantona varsinaisen LK-algoritmin toteutukselle. Yksinkertaisuus johtaa toisaalta siihen, että hyvin optimoitu 2-opt-algoritmi on erittäin nopea suorituskyvyltään, toisaalta se tuo esiin heurististen algoritmien keskeisen ongelman eli jumittumisen paikalliseen minimiin. 

### 6.2 LK-algoritmin mallitoteutus

_The Traveling Salesman Problem_-kirjan luku 15 _Tour Finding_ esittelee LK-algoritmin tavoitteet ja mallitoteutuksen. Mallitoteutuksessa pyritään lähes optimaaliseen ratkaisuun järkevässä suoritusajassa, mikä on myös tämän harjoitustyön tavoitteena. LK-algoritmin toteutuksesta on toisilla kirjoittajilla erilaisia painotuksia, kuten esimerkiksi nopeuden korostaminen ja tyytyminen vähemmän optimaalisiin ratkaisuihin. 

![Kahden kaaren vaihto](/images/figure_15_1.png)

Algoritmin idea perustuu polun kaarien vaihtoihin, joilla käännetään reitin osan solmujen järjestys päinvastaiseksi. Jos uuden reitin pituus on lyhyempi kuin alkuperäisen, säilytetään vaihto. Muuten vaihto hylätään. Yksinkertaisin vaihto on kahden kaaren vaihto, mutta yhtä hyvin voidaan vaihtaa kolme tai useampi kaari keskenään. Lisäksi kaaren vaihtoja voidaan tehdä peräkkäin useita ja hyväksyä, että reitin välivaiheen pituus on suurempi kuin alkuperäinen, jolloin on mahdollista löytää usean vaihdon tuloksena alkuperäistä lyhyempi reitti.  

LK-algoritmin suunnittelussa on panostettu erityisesti kaarien vaihtojen optimointiin. Haasteena kaarien vaihdossa on toisaalta se, että pitäisi välttää turhia kaarenvaihtoja, ja toisaalta etsiä rohkeasti lyhyempää reittiä eikä tyytyä paikallisesti optimaaliseen ratkaisuun. Käytännösssä LK-algoritmi toki lähes poikkeuksetta päätyy paikalliseen eikä yleiseen optimaaliseen ratkaisuun pois lukien hyvin yksinkertaiset verkot. Rekursiivisen haun leveyttä, eli kuinka monta naapurisolmua otetaan mukaan hakuun, rajoitetaan suhteessa haun syvyyteen. Mallitoteutuksessa syvyydelle ei ole asetettu ylärajaa, mutta käytännön toteutuksissa käytetään usein 10-12 ylärajaa rekursioille. Tyypillisesti haun leveys on rajoitettu ensimäisellä tasolla 5 naapuriin, toisella tasolla 5 naapuriin ja tätä seuraavilla tasoilla yhteen naapuriin.

#### Funktio `step()`

![Funktio step()](/images/algorithm_15_1.png)

Funktio `step()` on rekursiivinen funktio, joka etsii parannuksia nykyiseen kierrokseen suorittamalla mahdollisia k-opt-vaihtoja — yksi kerrallaan — niin kauan kuin parannus näyttää mahdolliselta.

Toiminta:
1. Luo lk-ordering eli lista lupaavista naapureista vertexille `base`.
2. Käy läpi korkeintaan `breadth(level)` lupaavaa naapuria (eli leveys tietyllä tasolla).
    - Jokaiselle kandidaatille yrittää tehdä flip-operaation, jolla kaksi reunaa korvataan kahdella uudella.
    - Jos flip parantaa kierrosta (tai saattaa johtaa parannukseen), funktio kutsuu itseään seuraavalla tasolla (depth-first-haku).
3. Jos jokin haara johtaa parempaan reittiin, se palautetaan. Muuten peruuntuu (backtrack) ja yrittää seuraavaa vaihtoehtoa.

#### Funktio `alternate_step()`

![Funktio alternate_step()](/images/algorithm_15_2.png)

Funktio `alternate_step()` laajentaa hakua etsimällä vaihtoehtoisia ensimmäisiä siirtoja ja syvempiä vaihtopolkuja, jotka eivät ole saatavilla step()-funktion kautta. Käyttää kolmitasoista leveyshakua (A-, B- ja D-tila) ja erilaisia flip-sekvenssejä.

Toiminnot:
1.	Luo A-ordering (lupaavat naapurit `next(base)`-solmulle).
2.	Jokaiselle a:
    - Luo B-ordering naapureille `next(a)`.
    - Jokaiselle b:
        - Jos b on reitillä `next(base)` -> a, suoritetaan vaihtoehtoinen flip-sarja (kuva 15.5).
        - Muuten: luodaan D-ordering ja suoritetaan syvempi flip (kuva 15.6).
3.	Jokainen flip-sarja tarkistaa, paraneeko kierros.
4.	Jos löytyy parannus, kutsutaan `step()`-funktiota seuraavalta tasolta.

#### Funktio `lk_search()` 

![Funktio lk_search()](/images/algorithm_15_3.png)

Funktio `lk_search()` äynnistää Lin–Kernighan-parannushaun yksittäisestä solmusta `v` ja annetusta kierroksesta `T`. Koostuu `step()`- ja `alternate_step()`-kutsujen ketjusta. Palauttaa parannetun flip-sekvenssin tai ilmoittaa, ettei parannusta löytynyt.

Toiminta:
1.	Alustaa:
    - Nykyinen kierros = `T`
    - Tyhjä flip-sekvenssi
    - Asettaa `base = v`
2.	Kutsuu:
    - `step(1, 0)` – yrittää löytää suoraa parannusta
    - Jos ei löydy, kutsuu `alternate_step()`
3.	Jos parannus löytyi, palauttaa flip-sekvenssin, muuten ilmoittaa epäonnistumisesta

#### Funktio `lin_kernighan()` 

![lin_kernighan()](/images/algorithm_15_4.png)

Funktio `lin_kernighan()` on pääfunktio, joka iteroi Lin–Kernighan-haun (`lk_search`) useilla aloitussolmuilla, ja päivittää parhaan tunnetun reitin, kunnes yhtään parannusta ei enää löydy. Se on koko heuristisen algoritmin moottori. 

Toiminta:
1.	Alustus:
    - Aseta `lk_tour = T` (lähtökierros).
    - Merkitse kaikki solmut aktiivisiksi (eli kelvollisiksi hakupisteiksi).
2.	Iteratiivinen parannushaku:
    - Niin kauan kuin löytyy merkittyjä solmuja:
        - Valitaan merkitty solmu `v`.
        - Kutsutaan `lk_search(v, lk_tour)`:
            - Jos löytyi parantava flip-sekvenssi:
            - Käydään flipit läpi yksi kerrallaan:
                - Sovelletaan flipiä `flip(x, y)`.
                - Päivitetään `lk_tour`.
                - Merkitään solmut `x` ja `y` (koska ne saattavat avata uusia parannuksia).
            - Poistetaan flip-sekvenssi listalta.
        - Jos parannusta ei löytynyt:
            - Poistetaan `v` aktiivisista solmuista.
3.	Palautus:
    - Lopuksi palautetaan `lk_tour`, eli paras löytynyt reitti.

#### Funktio `chained_lin_kernighan()`

![chained_lin_kernighan()](/images/algorithm_15_5.png)

Funktio `chained_lin_kernighan()` on edistyneempi versio `lin_kernighan()`-funktiosta. Sen tarkoituksena on jatkaa parannusten hakua myös sen jälkeen, kun tavallinen Lin–Kernighan ei enää löydä parannuksia — tekemällä hallittuja satunnaisia häiriöitä (kicks) reittiin ja käynnistämällä `lin_kernighan()` funktion uudelleen.

Toiminta:
1.	Alustus:
    - Suoritetaan tavallinen `lin_kernighan()`, josta saadaan alkuperäinen parannettu reitti T.
2.	Iteratiivinen ketjutus:
    - Niin kauan kuin aikaa on jäljellä:
        - Luodaan kick: flip-sekvenssi, joka häiritsee nykyistä reittiä T (yleensä 4-opt siirto kuten double-bridge).
        - Sovelletaan kick: T <- T'
        - Ajetaan `lin_kernighan(T')` -> saadaan uusi reitti Tʹ.
        - Jos Tʹ on parempi kuin alkuperäinen T, hyväksytään se:
            - T <- Tʹ
        - Muuten, palautetaan T takaisin vanhaan tilaansa (perutaan kick).
3.	Palautus:
    - Palautetaan paras löydetty reitti T.


## 7. Aika- ja tilavaativuus

Lin–Kernighan-heuristisen algoritmin aikavaativuus ja tilavaativuus ovat merkittävästi parempia kuin tarkkojen ratkaisualgoritmien, kuten Held-Karp-algoritmin. Vaikka Lin–Kernighan-algoritmille ei ole olemassa tiukkaa teoreettista aikarajaa, käytännön havaintojen perusteella sen suorituskyky sijoittuu useimmissa tapauksissa väliin $O(n^2 \cdot log(n))$ ja $O(n^3)$, riippuen muun muassa sallittujen vaihtoehtoisten reittivaihtojen määrästä, rekursion syvyydestä ja käytetyn naapurilistan pituudesta. Ketjutetussa versiossa (Chained Lin–Kernighan) aikavaativuus voi kasvaa hieman suuremmaksi, koska algoritmi suorittaa toistuvia häiriöitä (kick-vaiheita) ja käynnistää Lin–Kernighan-haun useita kertoja, mutta myös sen kohdalla kasvu pysyy käytännössä polynomisena.

Tilavaativuus on hallittavissa, sillä tärkeimmät muistia vievät komponentit ovat etäisyysmatriisi, jonka koko on $O(n^2)$, sekä rakenteet reitin esittämiseen ja naapurilistojen ylläpitämiseen, jotka vaativat tyypillisesti $O(n)$ – $O(k \cdot n)$ muistia. Koska k (esimerkiksi lähimmät 10-20 naapuria) on paljon pienempi kuin n, algoritmin muistinkäyttö pysyy maltillisena myös suurilla instansseilla. Tämä tekee Lin–Kernighan-heuristiikasta erittäin käyttökelpoisen erityisesti silloin, kun etsitään hyviä (ei-optimaalisia) ratkaisuja nopeasti suurissa TSP-ongelmissa. Toisin kuin eksponentiaalista aikaa vievät tarkat menetelmät, LK-algoritmi skaalautuu käytännössä tuhansiin solmuihin ja tuottaa laadukkaita reittejä kohtuullisessa ajassa ja muistinkulutuksessa. Näin se tarjoaa tasapainon tehokkuuden ja tuloksen laadun välillä.

| Algoritmi             | Aikavaativuus                | Tilavaativuus | Kuvaus |
|-----------------------|------------------------------|---------------|--------|
| Brute-force           | $(n-1)!$                     | $O(n)$        | Käy kaikki reitit läpi. |
| Held–Karp             | $O(n^2 \cdot 2^n)$           | $O(n \cdot 2^n)$  | Täsmällinen algoritmi dynaamisella ohjelmoinnilla. |
| Lin–Kernighan         | $O(n^2 \cdot log(n))$ – $O(n^3)$ | $O(n^2)$      | Heuristiikka, joka tekee k-opt-vaihtoja dynaamisesti. |
| Chained Lin–Kernighan | Hieman suurempi kuin LK      | $O(n^2)$      | Lisää satunnaisia häiriöitä ja toistaa LK-hakuja. |


## 8. Testausstrategia

Hyvä testausstrategia LK-algoritmin toteuttamisessa yhdistää toiminnallisen oikeellisuuden varmistamisen, heuristiikan laadun arvioinnin ja suorituskyvyn testauksen. Koska LK on heuristinen eikä yleensä palauta parasta ratkaisua, testaus ei voi perustua pelkästään oikean tuloksen vertailuun — vaan siihen, että ratkaisu on kelvollinen ja riittävän hyvä. 

### 8.1 Testauksen tavoitteet

Testauksen päätavoitteita ovat:
	1.	Oikeellisuus: varmista, että algoritmi palauttaa aina sallitun TSP-kierroksen eli se käy jokaisessa solmussa täsmälleen kerran ja palaa alkuun.
	2.	Laadukkuus: arvioi, kuinka hyvä reitti on verrattuna tunnettuun optimiin tai muihin algoritmeihin kuten 2-opt-algoritmiin.
	3.	Suorituskyky: mittaa algoritmin suoritusaika ja sen riippuvuus syötteen koosta.
	4.	Vikasietoisuus: testaa, että algoritmi toimii myös erikoistapauksissa eikä kaadu virheellisiin tai epätavallisiin syötteisiin.

### 8.2 Testitapaukset

1. Pienet käsin verifioitavat instanssit
    - Esimerkiksi 10–12 solmun TSP-ongelmat, joissa paras ratkaisu voidaan tarkistaa kaikki vaihtoehdot läpikäyvällä algoritmilla.
    - Näillä testataan rakenteellista oikeellisuutta ja mahdollisten flip-toimintojen virheettömyyttä.

2. Keskikokoiset tunnetut instanssit (esim. TSPLIB)
    - Testaa 40–300 solmun tapauksilla, joissa tunnetaan optimaalinen ratkaisu.
    - Vertaile LK:n ratkaisun pituutta optimiin tai muihin algoritmeihin.

3. Satunnaisesti generoituja instansseja
    - Luo euklidisia TSP-ongelmia esimerkiksi satunnaisilla pisteillä tasossa.
    - Näillä arvioidaan suorituskykyä ja skaalautuvuutta.

4. Erikoistapaukset
    - Kaikki kaupungit samalla etäisyydellä.
    - Kolmion muotoinen tai täysin lineaarinen järjestys.
    - Syötteet, joissa etäisyydet eivät noudata kolmioepäyhtälöä.

## 9. Rajapinnat ja integraatio

LK-algoritmia voidaan käyttää joko komentoriviltä ajamalla tiedosto sellaisenaan tai kirjastona tuomalla sen moduuliksi ja kutsumalla funktiota `chained_lin_kernighan(coords, init, time_limit)`. Algoritmi etsii oletuksena kansiossa `.../TSPLIB95/tsp/´ olevia TSP-tiedostoja, jotka täyttävät algoritmin vaatimukset. Omia TSP-tiedostoja voi suorittaa asettamalla ne omaan kansioonsa ja koodista kansion osoitteen.

## 10. Suoritusympäristö ja vaatimukset

LK-algoritmi on toteutettu Python 3.12.5 ohjelmointikiellä.  

Käytössä olevia Python standardikirjastoja ovat `os`, `itertools` ja `time`. Asennettavia Python kirjastoja ovat `NumPy` ja `matplotlib`. Lisäksi tarvitaan funktio `Delaunay` kirjastosta `SciPy.Spatial`.

Kansiosta `documentation` löytyy `pip freeze > requirements.txt` tehty lista kirjastoista versioineen.

## 11. Dokumentaation kieli

Harjoitustyön kirjallinen dokumentaatio on suomeksi. Solvelluksen lähdekoodin yhteydessä oleva dokumentaatio sekä versionhallinnan kommentit on englanniksi.

## 12. Lähteet

Applegate, David L. & Bixby, Robert E. & Chvtal,  Vaek & Cook, William J. (2006): *The Traveling Salesman Problem : A Computational Study*, Princeton University Press.

Lin, Shen & Kernighan, Brian W. (1973): ”An Effective Heuristic Algorithm for the Traveling-Salesman Problem”, Operations Research, Vol. 21, No. 2, s. 498–516.

Lin-Kernighan-Helsgaun algoritmin C-kielinen toteutus: http://webhotel4.ruc.dk/~keld/research/LKH/

Concorde TSP Solver on LK:n C-kielinen toteutus: https://www.math.uwaterloo.ca/tsp/concorde/gui/gui.htm
