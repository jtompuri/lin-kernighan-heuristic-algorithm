# Viikkoraportti 2

## 1. Mitä olen tehnyt tällä viikolla?

Olen kirjoittanut määrittelydokumenttia ja laatinut ensimmäiset versiot algoritmeista ja apufunktioista TSP-tiedostojen käsittelyyn ja kuvaajien piirtoon. Olen myös tehnyt ensimmäiset yksikkötestit LK-algoritmin Tour-luokan metodeille, joista keskeisin on filp-metodi, joka kääntää ympäri solmujen järjestyksen reitin kahden solmun rajaamalla osavälillä. Filp-metodin kanssa oli huomattavia ongelmia erityisesti rajatapauksissa, joissa osaväli alkaa listan lopusta ja jatkuu listan alussa. 

## 2. Miten ohjelma on edistynyt?

Olen tehnyt ensimmäiset versiot algoritmeista:
- Exact TSP solver on kaikki pienen verkon mahdolliset reitit läpikäyvä algoritmi
- Simple TSP solver on 2-opt-vaihtoja käyttävä yksinkertaistus LK-algoritmista
- Lin-Kernighan TSP solver on TSP-kirjan LK-algoritmin mallitoteutukseen perustuva algoritmi
- Lisäksi olen toteuttanut apufunktiot TSP-tiedostojen käsittelyyn ja kuvaajien piirtoon.

Olen toteuttanut mallitoteutuksen heuristiikkoja yksi kerrallaan varmistaen joka välissä, että algorimin tulos paranee tai ei ainakaan huonone. En ole vielä keskittynyt hakuparametrien hienosäätöön, vaan olen ottanut mallitoteutuksessa annetut arvot lähtökohdaksi.

## 3. Mitä opin tällä viikolla / tänään?

Olen miettinyt algoritmin oikeellisuuden tarkistamista, mikä heuristisessa algoritmissa tarkoittaa sitä, että päästään riittävän lähelle parasta ratkaisua. Koska heuristinen algoritmi ei tuota yleensä parasta ratkaisua, täytyy vertailukohdaksi ottaa varmasti toimiva algoritmi kuten Exact TSP solver tai julkisesti saatavilla olevaa testiaineisto, josta tunnetaan paras ratkaisu. Tähän TSPLIB on osoittautunut hyväksi työkaluksi. 

TSPLIB sisältää vajaa 40 TSP-ongelmaa, josta osaan on pystytty ratkasemaan tai todistamaan paras ratkaisu. Kaikki TSPLIB-ongelmat eivät ole symmetrisiä eikä niissä ole euklidista, kaksiuloitteista geometriaa, joten rajasin vertailun ongelmiin, jotka täyttävät määrittelydokumentin rajaukset. Tällaisia ongelmia on kaikkiaan 18 kappaletta ja verkkojen muoto poikkeaa toisistaan huomattavasti, joten testiaineisto on varsin kattava.     

Haluan tehdä itse oikeellisuus tarkistuksen pienillä verkoilla, joten toteutin Exact TSP solveriin verkon ja parhaan ratkaisun tallennuksen rand{n}.tsp ja rand{n}.opt.tour tiedostoiksi kansioon "/tsp", jolloin voin käyttää parhaita ratkaisuja Simple TSP solver ja Lin-Kernighan TSP solver -algoritmien testaamiseen 4-12 solmun satunnaisilla verkoilla.

## 4. Mikä jäi epäselväksi tai tuottanut vaikeuksia? 

Mainitsin edellisessä viikkoraportissa, että algoritmin toteutuksen kuvaus on TSP-kirjassa monimutkainen ja se jättää monia yksityiskohtia avoimeksi. Sain nyt kuitenkin toteutettua kaikki mallitoteutuksen heuristiikat, mutta minulle on vielä epäselvää, olenko ymmärtänyt toteutuksen oikein ja toimivatko heuristiikat niin kuin niiden tulisi. 

Keskeinen haaste LK-algoritmissa on välttää tai päästä ulos paikaillisista minimeistä, mihin lähes kaikki algoritmin heuristiikat pyrkivät. Lopputulokseen vaikuttaa merkittävästi hakuparametrien valinta: liian väljät parametrit johtavat haun eksloosioon, kun rekursio ei suppene vaan hajaantuu.   

## 5. Mitä teen seuraavaksi?

Seuraavia työvaiheita:
- LK-algoritmin parametrien testaaminen
- Toteutusdokumentin miettiminen ja kirjoittaminen
- Testisuunnitelman toteuttaminen ja tulosten raportointi
- Jos en ole tyytyväinen LK-algoritmiin suhteessa vertailualgoritmiin, täytyy lukea lisää aiheesta (esim. Keld Helsgaun)  

## Toisen viikon viikon tuloksia

Seuraavassa muutamia vertailuja algoritmien välillä. 

### Exact TSP solver vs. Simple TSP solver (4-12 solmua)

![Exact TSP solver vs. Simple TSP solver (4-12 solmua)](/images/simple_tsp_solver_plots_4_12_cities_20s.png)

Kun verkossa on solmuja 4-12 nähdään, että Simple TSP solver löytää kolmessa tapauksessa yhdeksästä parhaan ratkaisun. Kuvissa paras ratkaisu on kiinteä sininen viiva ja heuristinen ratkaisu oranssilla pisteviiva. Koska algoritmi päätyy hyvin nopeasti paikalliseen minimiin ja pysähtyy yleensä parissa sekunnissa, ei aika rajan kasvattaminen muuta tuloksia kuin jos sen asettaa hyvin pieneksi. Myöskään maksimi syvyydellä ei ollut merkitystä.

### Exact TSP solver vs. Lin-Kernighan TSP solver (4-12 solmua)

![Exact TSP solver vs. Lin-Kernighan TSP solver (4-12 solmua)](/images/lin_kernighan_tsp_solver_plots_4_12_cities_20s.png)

Kun verkossa on solmuja 4-12 nähdään, että myös LK-algorimi löytää kolmessa tapauksessa yhdekstästä parhaan ratkaisun. Kuvissa paras ratkaisu on kiinteä sininen viiva ja heuristinen ratkaisu oranssilla pisteviiva. Käytin aikarajana 5, 10 ja 20 sekuntia. Tuloksissa oli jonkin verran satunnaista vaihelua, mutta 10 ja 20 sekuntia tuotti keskimäärin paremman tuloksen. Seuraavissa kuvissa olen käyttänyt järjestelmällisesti 20 sekunnin aikarajaa.

### Simple TSP solver ja TSPLIB

![Simple TSP solver ja TSPLIB](/images/simple_tsp_solver_plots_20s.png)

Kuvassa nähdään TSP-ongelmien verkon muodon vaikuttavan ratkaisevasti approksimaation tarkkuuteen. Esimerkiksi ongelmassa `pr2392` näyttäisi löytyneen paras ratkaisu, kun taas ongelma `pcb442` tuottaa 93 %:n poikkeaman. Huomattavaa on kuitenkin, että tämä yksinkertainenkin algoritmi pääsee monessa ongelmassa 5-10 %:n päähän parasta ratkaisua, mikä voi olla ihan riittävä tarkkuus monessa käytännön sovelluksessa. Jos ajattelee vaikka ajoreittejä, joiden optimointiin vaikuttaa niin moni muu tekijä, niin tätä tarkemmalle arviolle tuskin on tarvetta ja kannattaa keskittyä muiden vaikuttavien tekijöiden arviointiin.

### Lin-Kernighan TSP solver ja TSPLIB

![Lin-Kernighan TSP solver ja TSPLIB](/images/lin_kernighan_tsp_solver_plots_20s.png)

Kuvaajista nähdään, että LK-algoritmi löytää useassa tapauksessa reitin, joka on alle 5 %:n päässä parhaasta ratkaisusta, mihin voi olla hyvin tyytyväinen. Pääsääntöisesti 20 sekunnin aikaraja tuotti paremman tuloksen kuin 5 sekunnin aikaraja, mutta mukana oli yksi poikkeus, joka voi toki olla sattumaa. Vaikeita ongelmia LK-algoritmille olivat `pcb442`, `pr1002` ja `pr76`. Huomataan, että LK-algoritmi pärjäsi keskimäärin selvästi paremmin kuin yksinkertainen algoritmi. Merkittävin ero oli `pcb442` ongelmassa. Toisaalta taas yksinkertainen algoritmi selvityi yllättäen paremmin ongelmasta `pr76`. Tämä voi olla sattumaa, sillä LK-algoritmi ratkaisi `pr76` ongelman 5,13 % poikkeamalla, kun aikaraja oli 5 sekuntia.

## Pohdintaa

TSPLIB:n ongelma ovat tunnettuja tieteellisissä artikkeleissa käytettyjä vaikeita TSP-ongelmia, joihin usein kyseinen artikkeli tarjoaa toimivaa heuristiikkaa ratkaisuksi. Esimerkiksi pr-alkuiset ongelmat ovat Manfred Padbergin ja Giovanni Rinaldin kehittämiä. Tutkijat ovat kehittäneet TSP-ongelmien ratkaisuun `Linear Programming solver'-algoritmeja, jotka oletettavasti toivat hyvin heidän esittelemien ongelmien ratkaisussa. 

Kun tutustuin LKH:n C-kieliseen lähdekoodiin, niin panin merkille, että se hyödyntää huomattavaa määrää erilaisia heuristiikkoja. Tämän harjoitustyön laajuudessa ei ole realistista ottaa esimerkiksi edellä mainittua LP-solver algoritmia mukaan. Voisin kuitenkin tutustua esimerkiksi ongelmaa `pr1002` koskevaan kirjallisuuteen. Jos siitä löytyy helposti toteutettava heuristiikka, joka auttaisi tuohon ongelmaan ja mahdollisesti myös muihin, niin sen sisällyttämistä LK-algoritmiin voi ainakin harkita.

## Suorituskyvystä

Testasin LK-algoritmin ajon yhteydessä Pythonin tarjoamaa sovelluksen suorituksen analysointia, jonka voi käynnistää komennolla `python -m cProfile profile.out oma_sripti.py`. Tallennetusta tilastosta sai yleiskuvan siitä, mihin sovelluksessa kuluu aikaa.  

```
Wed May 21 01:42:45 2025    profile.out

         413735069 function calls (411398460 primitive calls) in 107.529 seconds

   Ordered by: cumulative time
   List reduced from 4169 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    899/1    0.003    0.000  107.530  107.530 {built-in method builtins.exec}
        1    0.002    0.002  107.530  107.530 lin_kernighan_tsp_solver_8.py:1(<module>)
       18    0.028    0.002  106.524    5.918 lin_kernighan_tsp_solver_8.py:199(chained_lin_kernighan)
     2730    0.817    0.000  106.444    0.039 lin_kernighan_tsp_solver_8.py:173(lin_kernighan)
   581549    0.704    0.000   84.653    0.000 lin_kernighan_tsp_solver_8.py:164(lk_search)
2656082/581549   21.690    0.000   60.581    0.000 lin_kernighan_tsp_solver_8.py:86(step)
  5429692   34.007    0.000   42.954    0.000 lin_kernighan_tsp_solver_8.py:33(flip)
   954048    0.387    0.000   27.927    0.000 lin_kernighan_tsp_solver_8.py:72(tour_cost)
   954128   10.368    0.000   27.339    0.000 {built-in method builtins.sum}
150566920   16.970    0.000   16.970    0.000 lin_kernighan_tsp_solver_8.py:74(<genexpr>)
163250539    9.885    0.000    9.885    0.000 {method 'append' of 'list' objects}
   265113    0.802    0.000    3.143    0.000 lin_kernighan_tsp_solver_8.py:131(alternate_step)
 36401455    2.436    0.000    2.436    0.000 lin_kernighan_tsp_solver_8.py:27(prev)
   584297    2.037    0.000    2.075    0.000 lin_kernighan_tsp_solver_8.py:21(__init__)
 25292170    1.768    0.000    1.768    0.000 lin_kernighan_tsp_solver_8.py:26(next)
  3020553    1.122    0.000    1.586    0.000 {method 'sort' of 'list' objects}
     2730    0.981    0.000    1.265    0.000 lin_kernighan_tsp_solver_8.py:77(delaunay_neighbors)
      103    0.002    0.000    1.125    0.011 __init__.py:1(<module>)
     2748    0.417    0.000    0.780    0.000 lin_kernighan_tsp_solver_8.py:69(build_distance_matrix)
    978/4    0.002    0.000    0.545    0.136 <frozen importlib._bootstrap>:1349(_find_and_load)
```