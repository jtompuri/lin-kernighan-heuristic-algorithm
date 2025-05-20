# Viikkoraportti 1

## 1. Mitä olen tehnyt tällä viikolla?

Käytin aikaa aiheen valintaan Eigenface-kasvojentunnistuksen ja Lin-Kernighan algoritmin välillä.

Yritin toteuttaa Eigenfaces-algoritmin. Skaalasin ja rajasin Yalefaces-kirjaston 16 380 kuvaa Matlabissa. laskin eigenface-matriisin. Tein matriisille pääkomponenttianalyysin. Laskin 95% tarkkuuteen riittävän määrän pääkomponentteja, joka oli reilu 200 dimensiota. Siirsin aineiston Pythoniin ja toteutin kasvontunnistusalgoritmin, jolla yritin tunnistaa, onko kuvassa kasvoja. Tämä ei toiminut, sillä algoritmi tunnisti tasaisissa pinnoissa kasvoja, mikä johtui luultavasti siitä, että eigenfaces toimii pikemminkin henkilön kasvojen tunnistamisessa muista henkilöistä kuin kasvojen tunnistamiseen kuvasta. Tutkin asiaa ja kasvojentunnistukseen kuvasta on olemassa paljon parempia algoritmeja, joten hylkäsin aiheen.

Valitsin aiheeksi Lin-Kernighan heuristisen algoritmin. Luin alkuperäisen artikkelin vuodelta 1973 ja lainasin ja kuin kirjan _The Traveling Salesman Problem: Computational Study_ luvut TSP:n taustasta ja Lin-Kernighan algoritmista sekä Concorde TSP Solver -ohjelmasta, joka tekijöitä kirjan kirjoittajat ovat. Latasin verkosta ja käänsin Concorden C-kielisen lähdekoodin ja testasin sitä luomiini testiaineistoihin ja TSPLIB-aineistoihin. Latasin ja käänsin myös LKH TSP Solver ohjelman, joka on tällä hetkellä tehokkain heuristinen TSP-ratkaisija. Tutustuin kursorisesti C-kieliseen lähdekoodiin. 

Pohdin aihetta ja kävin keskustelun Hannu Kärnän kanssa, jonka pohjalta päätin työn rajauksen. Dokumentoin suunnitelman projektin Github-sivulle.    

## 2. Miten ohjelma on edistynyt?

LK-algoritmi perustuu sitä edeltäneeseen opt2-algoritmiin, joten olen tehnyt kokeiluja yksinkertaisemalla opt2-algoritmilla, joka toimii yllättävän hyvin pienillä optimoinneilla. Ilman optimointeja sain ratkaistua noin 50 solmun verkon ja ottamalla käyttöön NumPyn taulukon tietorakenteeksi ja nopeuttamalla silmukoita kääntämällä ne konekielisiksi numban JIT-käännöstyökalulla, sain ratkaistua 300 solmun verkon. Tätä suuremmilla verkoilla algoritmi luultavasti jäi jumiin paikalliseen minimiin ja Python ikuiseen luuppiin. Tarkoitus on toteuttaa LK-algoritmin "variable k-opt"-algoritmin ja vältettyä paikalliset minimit esimerkiksi "double bridge kick"-algoritmilla. 

## 3. Mitä opin tällä viikolla / tänään?

Perusideat LK-algoritmissa ja TSP:n heuristisessa ratkaisemisessa. Olen kerrannut kerrattua TSP-algoritmien aikavaativuuksia. Algoritmit opt-2, opt-3 ja variable k-opt. 

## 4. Mikä jäi epäselväksi tai tuottanut vaikeuksia? 

Alkuperäinen artikkeli on vaikeaselkoinen. Onneksi TSP-kirja on huomattavasti selkeämpi ja myös nostaa esiin alkuperäisen artikkelin algoritmin puutteita. Minulle on syntynyt yleiskuva variable k-opt -algoritmista, mutta lähteiden pseudokoodi jättää toteutukseen tosi paljon kysymyksiä esimerkiksi valittavista tietorakenteista. Kirjassa kyllä puhutaan näistä, kuten linkitetyistä listoista ja kaksitasoisista linkitetyistä listoista, mutta toteutuksessa pitää huomioida Pythonin asettamat rajoitukset. Kirja puhuu myös monista LK-algoritmiin tehdyistä parannuksista Concorde-sovellusta kehitettäessä. Jotkut näistä parannuksista voisi olla toteutettavissa tässä projektissa, mutta osa menee selvästi yli realistisesta työn rajauksesta, kuten esimerkiksi "QSopt linear programming solver"-kirjaston hyödyntäminen algoritmissa.

## 5. Mitä teen seuraavaksi?

Työ jakautuu karkeasti näihin vaiheisiin:
- LK-algoritmin tutkiminen kirjallisuudessa ja ohjelmointi
  - Alustava algoritmi jää paikalliseen minimiin, joten se ei tuota parasta ratkaisua
  - Lasketaan brute forcella paras ratkaisu pieneen verkkoon ja verrataan luonnoksen tulokseen    
  - Tarkoitus on supistaa rekursiivisen haun leveyttä rekursion syvyyden kasvaessa
- testauksen suunnittelu ja toteuttaminen
- Brute force ja opt-2 algoritmien toteutus vertailua varten

## Ensimmäisen viikon tuloksia

Tiedostossa recursive-k-opt-lin-kernighan.py löytyy alustava luonnos LK-algoritmista, joka pystyy ratkaisemaan järkevässä ajassa satojen solmujen verkon likimääräisen lyhimmän reitin. Algoritmin tuottamista kuvaajista näkee, että reitti ei ole optimaalinen, sillä riippuen sattunaisluvun siemenarvosta reitti tekee isoja hyppäyksiä tai se kulkee poikki oman reittinsä. On tunnetusti todistettu, että optimaalinen reitti ei koskaan risteä itsensä kanssa, joten reitti ei voi olla optimaalinen. Tarkoitus on todistaa tämä pienillä verkoilla käyttämällä parhaan tuloksen antavaa, raakaan laskentaan perustuvaa algoritmia. 

### Satunnainen 100 solmun polku

LK-algoritmin luonnoksessa valitaan lähtökohdaksi satunnainen reitti. Lähtökohtana voisi sen sijaan olla jollain tehokkaalla algoritmilla, kuten "Nearest Neighbor"-algoritmilla (NN), laskettu approksimaatio, jota sitten tarkennetaan LK:lla. Hieman yllättäen TSP-kirjan mukaan satunnaisen ja NN-algoritmilla luodun verkon ratkaisun kestossa tai tarkkuudessa ei ole merkittävää eroa. NN-algoritmia itse asiassa käytetään LK:ssa, kun valitaan paikallisesti kaaria, joita ristiin kytkemällä etsitään lyhyempää reittiä, sillä kaaret valitaan alkaen lähimmistä solmuista edeten kaukaisempiin solmuihin. 

![Satunnainen 100 solmun polku](/kuvat/random_tour.png)

### LK:n ratkaisu 100 solmun polulle (ei optimaalinen)

Tässä LK-algoritmin luonnoksen ei-optimaalinen ratkaisu 100 solmun polulle. Kuten nähdään, niin se ei missään vaiheessa päädy kokeilemaan kaarien vaihtoa "pullistumalle", joka rajautuu kaarella, joka ylittää reitin kahdesti. Toimiessaan oikein rekursion pitäisi käydä läpi myös vaihtoehto, että kaikki "pullistuman" kaaret korvataan niin, että reitti ei enää risteä itsensä kanssa. Toisin sanoen, algoritmi jää jumiin paikalliseen minimiin. Jotta pullistuma poistettaisiin, algoritmin tulisi kokeilla rohkeasti pullistuman kaikkien 7 kaaren vaihtamista. Kuten nähdään, pullistuman solmujen muodostama reitti on paikallisesti optimaalinen, sillä kaaret muodostavat lyhimmän reitin solmujen välillä; on tunnetusti todistettu, että "kumilanka" konveksin kehän ympäri muodostaa aina lyhimmän reitin. Kehittyneemmissä LK-algoritmeissa, kuten alkuperäisen artikkelin ja TSP-kirjan LK-algoritmin esittelyssä, käytetään paikallisten minimien välttämiseen "kick"-operaatiota, jolla tuotetaan polkuun satunnaisia "double bridge"-rakenteita, jotka kääntävät polun ristiin kahdesta kohdasta ja näin tilapäisesti pidentävät polkua. "Kick"-operaation tarkoitus on pakottaa algoritmi ulos paikallisesta minimistä. Jos "kick"-operaation tuloksena löydetään sitä edeltänyttä lyhyempi reitti, niin algoritmi jatkaa tästä. Muussa tapauksessa "kick"-operaatio perutaan. 

![LK:n ratkaisema 100 solmun polku](/kuvat/lk-k-depth-1.png)

### Animaatio 20 solmun ratkaisusta (ei optimaalinen)

Animaatio siitä miten luonnos LK-algoritmista lyhentää iteratiivisesti polun pituutta vaihtamalla kaaria ristiin. LK pystyy vaihtamaan enemmän kuin kaksi kaarta keskenään, mutta nämä useamman kaaren vaihdot perustuvat ketjutettuihin kahden kaaren vaihtoihin, minkä vuoksi LK-algoritmin ytimessä on opt-2-algoritmi, joka vaihtaa kaksi kaarta keskenään. Animaatio ei esitä hylttyjä vaihtoehtoja, jotka osoittautuivat pidemmiksi, eikä siinä näy kaarien rekursiivisen haun syvyys. Jatkossa voisin raportoida mahdollisesti myös rekursion syvyyden. Yllätteän LK:n luonnoksessa en saanut ratkaisussa tuotettua eroja eri sallituilla maksimi rekursion syvyyksillä. Luonnollisesti, kun rekursio syvyys on nolla (max_k = 0), niin algoritmi tuottaa satunnaisen polun (ks. yllä), mutta k:n arvoilla 1-5 sain kaikilla saman ratkaisun 100 solmun verkolle (ks. yllä). Eroa oli lähinnä algoritmin suoritusajassa: kun k = 1, aikaa kului puolet ajasta verrattuna, kun k = 5. Pitää tutkia, onko algoritmissa jotain vikaa vai liittyykö tämä verkon muotoon. 

![Animaatio 20 solmun ratkaisusta](/kuvat/lk_tsp.gif)

Animaation viimeinen ruutu, johon LK-algoritmin luonnos päättyy:

![Animaation viimeinen ruutu](/kuvat/animation.png)


