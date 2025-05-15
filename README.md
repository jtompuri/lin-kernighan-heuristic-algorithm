# Lin-Kernighan heuristinen algoritmi

Lin-Kernighan heuristinen algoritmi (_Lin-Kernighan Heuristic Algorithm_) on tehokas algoritmi symmetrisen kauppamatkustajan ongelman (_Traveling Salesperson Problem_) likimääräiseen ratkaisuun keskimääräisellä aikavaativuudella $O(n^2)$. Lin-Kernighan heuristinen algoritmi (jatkossa LK) ei anna aina parasta ratkaisua ja se voi jäädä jumiin paikalliseen minimiin. LK esiteltiin Shen Linin ja Brian W. Kernighanin artikkelissa ”An Effective Heuristic Algorithm for the Traveling-Salesman Problem” vuonna 1973[^1]. LK ja sen ideoihin perustuvat algoritmit ovat edelleen tehokkaimpia likimääräisiä ratkaisuja kauppamatkustajan ongelma (jatkossa TSP). LK:ta myös käytetään alkuratkaisuna algoritmeille, jotka pystyvät löytämään parhaan ratkaisun TSP:lle[^2]. 

## Toteutussuunnitelma
Tämä on kurssityö Helsingin yliopiston Tietojenkäsittelytieteen laitoksen kurssille _Tekoäly ja algoritmit_ (5-6/2025). Työn tavoitteena on toteuttaa vertailualgoritmeja selvästi tehokkaampi algoritmi ja säilyttää ratkaisun laatu hyvänä, mikä tutkimuskirjallisuuden perusteella on usein alle 2 %:n poikkeama parhaasta ratkaisusta. Aloitan työn laatimalla yksinkertaisen LK:n pääpiirteet toteuttavan algoritmin Pythonilla. Laadin lisäksi yksinkertaisen kaikkien vaihtoehtojen läpikäyntiin perustuvan parhaan ratkaisun  aikavaativuudella *O(n!)* löytävän algoritmin. Lisäksi laadin Held-Karp-algoritmin, joka perustuu dynaamiseen ohjelmointiin ja löytää parhaan ratkaisun aikavaatimuksella $O(n^2*2^n)$. TSP on tunnetusti päätösongelmana vaativuusluokaltaan NP-täydellinen, kun kysytään onko verkossa hamiltonin polkua. Optimointiongelma TSP on NP-kova, kun halutaan selvittää verkon lyhin polku. 

Alkuperäinen Linin ja Kernighanin artikkeli[^1] kuvaa LK:n kombinatorisen heuristiikan satunnaisen ratkaisun iteratiiviseksi parantamiseksi seuraavasti:
1. Luo pseudosatunnainen ratkaisu T, joka täyttää kriteerit
2. Yritä löytää parannettu ratkaisu T’ jollain T:n muunnoksella
3. Jos ratkaisu T’ on parempi, niin T = T’ ja palaa kohtaan 2.
4. Jos parempaa ratkaisua ei löydy, niin T on paikallinen paras ratkaisu.

Artikkeli ei käytä vaikiintunutta pseudokoodia ja alkuperäisessä algoritmin kuvauksessa on myöhemmin tunnistettuja puutteita, jotka vaikuttavat suorituskykyyn suuremmilla syötteillä. Tämän vuoksi käytän ensisijaisena lähteenä LK:n toteutukselle TSP:tä käsittelevän *The Traveling Salesman Problem: A Computational Study*[^3] kirjan esitystä LK:sta. Teoksessa käydään myös läpi eri parannuksia LK:iin, joita pyrin toteuttamaan Pythonin asettamissa rajoissa asteittain kehitysversioissa.

## Testaus
Laadin testitapauksia eri syötteillä, joilla voidaan verrata LK:n suorituskykyä ja approksimaation laatua suhteessa vertailukohtana käytettäviin algoritmeihin, jotka antavat aina parhaan ratkaisun. Syötteenä käytän eri kokoisia ja eri muotoisia verkkoja. Tutkimuskirjallisuudessa käytetään TSP-tiedostoja kuvaamaan verkkoja, joten toteutan testisyötteet TSP-muodossa. Tällöin voin myös käyttää valmiita testisyötteitä.

Testauksessa minua kiinnostaa, miten algoritmi suoriutuu ei-satunnaisesti jakautuneesta verkosta. Hyvin tyypillisesti testiverkko luodaan satunnaisesti, mutta jos ajatellaan esimerkiksi kaupunkien sijoittumista maantieteellisesti, niin niiden sijaintiin vaikuttaa ratkaisevasti luonnolliset maantieteelliset esteet, kuten vesistöt ja vuoristot. Tästä seuraa, että kaupunkeja sijaitsee historiallisesti tiheämmin vesistöjen rannoilla ja harvemmin vuoristoissa eikä lainkaan vesistöissä.

Hyödynnän testausta LK:n parantamiseen asteittain kirjallisuudessa ehdotetuilla parannuksilla algoritmiin ja sen parametreihin ja tietorakenteisiin. Esimerkkinä parametrien valinnasta on dynaamisessa *k-opt*-algoritmissa käytettävä haun leveys eli $k$ eri haun syvyyksillä. LK eroaa sitä edeltäneistä algoritmeista, jotka käyttävät verkon kaarien vaihtamiseen *2-opt*- ja *3-opt*-algoritmeja, että LK käyttää vaihtuvaa leveyttä eri haun syvyyksillä etsiessään parasta ratkaisua. 

## Ohjelmointikieli
Kurssin suositeltu ohjelmointikieli on Python, joten käytän sitä LK:n toteuttamiseen, mikä asettaa rajoituksia LK:n suorituskyvylle. Tulkittuna kielenä Python jää suorituskyvyssä merkittävästi jälkeen käännetyistä kielistä kuten C/C++-kielistä. Pythonista myös puuttuu tehokkaita silmukka- ja tietorakenteita, joiden avulla LK:n suorituskykyä voitaisiin parantaa merkittävästi. Toisaalta ohjelmointikielen rajoitukset eivät vaikuta tulosten laatuun. Kurssin oppimistavoitteita on algoritmien opiskelu ja toteuttaminen, mihin Python soveltuu hyvin. 

Python toteutus LK:sta pystyy luultavasti ratkaisemaan  järkevässä ajassa kokoluokaltaan satojen solmujen verkkoja, mutta ei tuhansien tai kymmenien tuhansien solmujen verkkoja, kuten parhaat C-kieliset toteutukset *LKH*[^4] ja *Concorde*[^5]. Vertailun vuoksi *LKH* löysi 10 000 solmun verkon parhaan ratkaisun 17 minuutissa 15 sekunnissa tietokoneellani. 

Pythonin rajoitteet saattavat algoritmin suorituskyvyn testauksessa peittää joidenkin parannusten hyödyt. Oletan näin käyvän mahdollisesti niin sanotun ketjutetun LK:n osalta (_Chained Lin-Kernighan Algorithm_). Pyrin mahdollisuuksien mukaan raportoimaan myös Python rajoitusten vaikutuksia suorituskykyyn ja tulosten laatuun.

## Raportointi
Säilytän toimivat ja testatut LK:n kehitysversiot ja niiden testitulokset. Raportoin loppulisen LK:n tulosten lisäksi aikasarjana LK:n suorituskyvyn asteittaisen kehityksen tehtyjen parannuksen myötä. Raportoin mahdollisuuksien mukaan myös epäonnistuneita kokeiluja ja tuloksiltaan heikompia saman kehitysvaiheen vaihtoehtoja. 

LK:n ajonaikana tulostetaan tietoa laskennan etenemisestä ja lopuksi raportoidaan tilastotietoja laskennasta, kuten suoritusaika, tulosten laatu (suhdeluku tulos / paras tulos) ja suoritettujen askelten määrä. Yksikkötestejä ja ajonaikaista lokitulostusta käytetään tarpeen mukaan ohjelman virheiden selvittämiseen.

[^1]: Lin, Shen & Kernighan, Brian W. (1973): ”An Effective Heuristic Algorithm for the Traveling-Salesman Problem”, Operations Research, Vol. 21, No. 2, s. 498–516.

[^2]: Mulder, Samuel A. & Wunsch II, Donald C. (2003): ”Million city traveling salesman problem solution by divide and conquer clustering with adaptive resonance neural networks”, Neural Networks 16, s. 827–832.

[^3]: Applegate, David L. & Bixby, Robert E. & Chvtal,  Vaek & Cook, William J. (2006): *The Traveling Salesman Problem : A Computational Study*, Princeton University Press.

[^4]: Lin-Kernighan-Helsgaun algoritmin C-kielinen toteutus: http://webhotel4.ruc.dk/~keld/research/LKH/

[^5]: Concorde TSP Solver on LK:n C-kielinen toteutus: https://www.math.uwaterloo.ca/tsp/concorde/gui/gui.htm
