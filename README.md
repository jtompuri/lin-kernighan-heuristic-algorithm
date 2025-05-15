# Lin-Kernighan heuristinen algoritmi

Linin-Kernighanin heuristinen algoritmi (Lin-Kernighan Heuristic Algorithm) on tehokas algoritmi symmetrisen kauppamatkustajan ongelman likimääräiseen ratkaisuun aikavaativuudella $~O(n^2)$. Linin-Kernighanin heuristinen algoritmi (jatkossa LK) ei anna aina parasta ratkaisua ja se voi jäädä jumiin paikalliseen minimiin. LK esiteltiin Shen Lin ja Brian W. Kernighanin artikkelissa ”An Effective Heuristic Algorithm for the Traveling-Salesman Problem” vuonna 1973[^1]. LK ja sen ideoihin perustuvat algoritmit ovat edelleen tehokkaimpia likimääräisiä ratkaisuja kauppamatkustajan ongelmaan ja LK:ta käytetään alkuratkaisuna algoritmeille, jotka pystyvät löytämään parhaan ratkaisun[^2]. Tämä on kurssityö Helsingin yliopiston Tietojenkäsittelytieteen laitoksen kurssille _Tekoäly ja algoritmit_ (5-6/2025).    

## Toteutussuunnitelma
Aloitetaan työ laatimalla mahdollisimman yksinkertainen LK:n pääpiirteet toteuttava algoritmi Pythonilla. Laaditaan lisäksi suhteellisen yksinkertainen, mutta ei kaikkien vaihtoehtojen läpikäyntiin perustuva (aikavaatimus $O(n!)$), parhaan ratkaisun löytävä algoritmi. Tällainen voisi olla esimerkiksi Held-Karp-algoritmi, joka perustuu dynaamiseen ohjelmointiin ja löytää parhaan ratkaisun aikavaatimuksella $O(n^2*2^n)$. 

Alkuperäinen artikkeli[^1] kuvaa LK:n kombinatorisen heuristiikan yleisellä tasolla satunnaisen ratkaisun iteratiiviseksi parantamiseksi seuraavasti:
1. Luo pseudosatunnainen ratkaisu T, joka täyttää kriteerit
2. Yritä löytää parannettu ratkaisu T’ jollain T:n muunnoksella
3. Jos ratkaisu T’ on parempi, niin T = T’ ja palaa kohtaan 2.
4. Jos parempaa ratkaisua ei löydy, niin T on paikallinen paras ratkaisu.

Alkuperäinen artikkeli ei käytä modernia pseudokoodia ja alkuperäisessä esityksessä on myöhemmin tunnistettuja puutteita, jotka vaikuttavat suorituskykyyn suuremmilla syötteillä. Tämän vuoksi käytän ensisijaisena lähteenä LK:n toteutukselle teoksen *The Traveling Salesman Problem: A Computational Study*[^3] esitystä LK:sta. Teoksessa käydään  läpi eri parannuksia LK:iin, jotka pyrin toteuttamaan Pythonin asettamissa rajoissa asteittain kehitysversioissa.

## Testaus kehityksen aikana
Laaditaan testitapauksia eri syötteillä, joilla voidaan verrata LK:n suorituskykyä suhteessa vertailukohtana käytettävään algoritmiin. Syötteenä voidaan käyttää eri kokoisia ja eri muotoisia verkkoja. Tutkimuskirjallisuudessa käytetään TSP-tiedostoja kuvaamaan verkkoja, joten toteutetaan testisyötteet tässä muodossa. Tällöin voidaan käyttää kattavia valmiita testisyötteitä. 

Testauksessa minua kiinnostaa erityisesti, miten algoritmi suoriutuu ei-satunnaisesti jakautuneesta verkosta. Hyvin tyypillisesti testiverkko luodaan satunnaisesti, mutta jos ajatellaan kaupunkien sijoittumista maantieteellisesti, niin niiden sijaintiin vaikuttaa ratkaisevat luonnolliset maantieteelliset esteet, kuten vesistöt ja vuoristot. Tästä seuraa, että esimerkiksi kaupunkeja sijaitsee tiheämmin merten ja jokien rannoilla ja harvemmin vuoristossa.

Testausta on tarkoitus hyödyntää LK:n parantamiseen asteittain kirjallisuudessa ehdotetuilla parannuksilla algoritmiin ja sen parametreihin ja tietorakenteisiin. Esimerkkinä parametrien valinnasta on dynaamisessa *k-opt*-algoritmissa käytettävä haun leveys, kun arvo *k* eli haun syvyys kasvaa. LK:n erottaa edeltäneistä algoritmeista, jotka käyttävät *2-opt*- ja *3-opt*-algoritmeja, siinä että LK voi tehdä *k*-syvyisen haun. 

## Ohjelmointikieli
Kurssin suositeltu ohjelmointikieli on Python, joten käytän sitä LK:n toteuttamiseen, mikä asettaa suorituskyvylle rajoituksia. Tulkittuna kielenä Python jää suorituskyvyssä merkittävästi jälkeen käännetyistä kielistä kuten C/C++-kielistä. Pythonista myös puuttuu tehokkaita silmukka- ja tietorakenteita, joiden avulla LK:n suorituskykyä voitaisiin parantaa merkittävästi. 

Python toteutus LK:sta pystyy luultavasti ratkaisemaan kokoluokaltaan satojen solmujen verkkoja, mutta ei tuhansien tai kymmenien tuhansien solmujen verkkoja, kuten parhaat C-kieliset toteutukset *LKH*[^4] ja *Concorde TSP Solver*[^5]. Vertailun vuoksi *LKH* löysi 10 000 solmun verkon parhaan ratkaisun reilussa 17 minuutissa 15 sekunnissa tietokoneellani. Ohjelmointikielen rajoitukset eivät vaikuta kurssin oppimistavoitteisiin, joita ovat algoritmien opiskelu ja toteuttaminen. 

Pythonin rajoitteet saattavat algoritmin suorituskyvyn testauksessa peittää joidenkin parannusten vaikutuksen. Oletan näin käyvän esimerkiksi niin sanotun ketjutetun LK:n osalta (Chained Lin-Kernighan Algorithm). Pyrin mahdollisuuksien mukaan raportoimaan myös Python rajoitusten vaikutuksia.

## Raportointi
Säilytän toimivat ja testatut LK:n kehitysversiot ja niiden testitulokset. Raportoin loppulisen LK:n tulosten lisäksi aikasarjana LK:n suorituskyvyn asteittaisen kehityksen tehtyjen parannuksen myötä. Raportoin mahdollisuuksien mukaan myös epäonnistuneita kokeiluja ja tuloksiltaan heikompia saman kehitysvaiheen vaihtoehtoja.

LK:n ajonaikana tulostetaan tietoa laskennan etenemisestä ja lopuksi raportoidaan tilastotietoja laskennasta, kuten suoritusaika ja suoritettujen askelten määrä. Ajonaikaista tulostusta käytetään myös ohjelman virheiden selvittämiseen.

## Lähteet:
[^1]: Lin, Shen & Kernighan, Brian W. (1973): ”An Effective Heuristic Algorithm for the Traveling-Salesman Problem”, Operations Research, Vol. 21, No. 2, s. 498–516.

[^2]: Mulder, Samuel A. & Wunsch II, Donald C. (2003): ”Million city traveling salesman problem solution by divide and conquer clustering with adaptive resonance neural networks”, Neural Networks 16, s. 827–832.

[^3]: Applegate, David L. & Bixby, Robert E. & Chvtal,  Vaek & Cook, William J. (2006): *The Traveling Salesman Problem : A Computational Study*, Princeton University Press.

[^4]: Lin-Kernighan-Helsgaun algoritmin C-kielinen toteutus: http://webhotel4.ruc.dk/~keld/research/LKH/

[^5]: Concorde TSP Solver on LK:n C-kielinen toteutus: https://www.math.uwaterloo.ca/tsp/concorde/gui/gui.htm
