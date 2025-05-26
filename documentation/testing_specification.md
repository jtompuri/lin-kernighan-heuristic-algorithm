# Testausdokumentti

Tämä dokumentti kuvaa Lin-Kernighan TSP-ratkaisijan Python-toteutuksen testausprosessia. Testaus on suoritettu pääasiassa automaattisilla yksikkötesteillä `pytest`-kehystä hyödyntäen.

## 1. Yksikkötestauksen kattavuusraportti

Yksikkötestauksen kattavuusraportti tuotettaisiin tyypillisesti erillisellä työkalulla, kuten `coverage.py`. Suorittamalla esimerkiksi komento `coverage run -m pytest && coverage report -m` projektin juuressa saataisiin yksityiskohtainen raportti siitä, mitkä koodirivit ja haarat testit kattavat. Tämän dokumentin puitteissa oletetaan, että tällainen raportti on ajettu ja sen tuloksia hyödynnetään testauksen kehittämisessä. Tavoitteena on mahdollisimman korkea testikattavuus kriittisille osille.

## 2. Mitä on testattu, miten tämä tehtiin?

Ohjelman eri komponentteja on testattu seuraavasti:

**A. `Tour`-luokka (`test_tour.py`)**

`Tour`-luokka, joka edustaa kierrosta ja sen operaatioita, on testattu kattavasti.
*   **Alustus ja `get_tour()`:** Testattu, että `Tour`-olio alustuu oikein annetulla solmujärjestyksellä ja että `get_tour()`-metodi palauttaa tämän järjestyksen, normalisoiden sen alkamaan solmusta 0, jos se on kierroksessa mukana. Esimerkiksi syötteellä `[1, 0, 2]` `get_tour()` palauttaa `[0, 2, 1]`.
*   **`next()` ja `prev()`:** Varmistettu, että metodit palauttavat oikean seuraajan ja edeltäjän kierroksella eri solmuille. Esimerkiksi kierroksella `[0, 1, 2]`, `tour.next(2)` palauttaa `0` ja `tour.prev(0)` palauttaa `2`. Testattu myös tyhjän kierroksen ja virheellisten syötteiden (solmu ei kierroksessa) aiheuttamat `IndexError`-poikkeukset.
*   **`sequence()`:** Testattu, tunnistaako metodi oikein, onko tietty solmu annetulla kierroksen segmentillä. Esimerkiksi kierroksella `[0, 1, 2, 3, 4]`, `tour.sequence(0, 2, 4)` palauttaa `True`, kun taas `tour.sequence(3, 1, 0)` (ylittävä segmentti) palauttaa `True` solmulle `1`.
*   **`flip()`:** Testattu segmentin kääntämisen oikeellisuus. Esimerkiksi kierroksella `[0, 1, 2, 3, 4]`, `tour.flip(1, 3)` muuttaa kierroksen muotoon `[0, 3, 2, 1, 4]`. Varmistettu, että sisäinen `order`-taulukko ja `pos`-taulukko päivittyvät oikein.
*   **Kustannusten laskenta (`__init__` ja `init_cost`):** Testattu, että kierroksen kustannus lasketaan oikein. Esimerkiksi kolmion solmuilla `(0,0), (3,0), (0,4)` ja kierroksella `[0,1,2]`, kustannus on `3 + 5 + 4 = 12`. Myös tyhjien ja yhden solmun kierrosten kustannukset (0.0) on varmistettu.
*   **`flip_and_update_cost()`:** Testattu, että metodi kääntää segmentin oikein ja päivittää kierroksen kokonaiskustannuksen oikein. Esimerkiksi kierroksella `[0,1,2,3]` ja etäisyyksillä `D[0,1]=1, D[1,2]=1, D[2,3]=1, D[3,0]=1` (kustannus 4), `flip_and_update_cost(1,2)` (kääntää `[1,2]` -> `[2,1]`) muuttaa kierroksen `[0,2,1,3]`. Jos `D[0,2]=1.5, D[1,3]=1.5, D[0,1]=1, D[2,3]=1`, niin vanhat reunat `(0,1)` ja `(2,3)` (summa 2) korvautuvat reunoilla `(0,2)` ja `(1,3)` (summa 3). Kustannusmuutos on `+1`, ja uusi kustannus on `5`.

**B. Aputoiminnot (`test_utils.py`)**

Moduulin keskeiset aputoiminnot on testattu.
*   **`build_distance_matrix()`:** Testattu etäisyysmatriisin laskenta. Esimerkiksi koordinaateilla `[[0,0], [3,0], [0,4]]`, matriisin tulee olla `[[0,3,4], [3,0,5], [4,5,0]]` (pyöristettynä). Myös tyhjä ja yhden solmun tapaukset on testattu.
*   **`delaunay_neighbors()`:** Testattu Delaunay-naapurien löytäminen. Esimerkiksi neliölle `[[0,0],[1,0],[1,1],[0,1]]` solmun `0` naapurit ovat `1` ja `3`.
*   **`double_bridge()`:** Testattu "double bridge" -potkutoimintoa. Esimerkiksi kierroksella `[0,1,2,3,4,5,6,7]`, jos satunnaisesti valitut leikkauspisteet ovat `p1=1, p2=3, p3=5, p4=7`, segmentit ovat `s0=[0]`, `s1=[1,2]`, `s2=[3,4]`, `s3=[5,6]`, `s4=[7]`. Uusi kierros on `s0+s2+s1+s3+s4 = [0,3,4,1,2,5,6,7]`.
*   **`read_opt_tour()`:** Testattu `.opt.tour`-tiedostojen lukeminen. Esimerkiksi tiedostosisällöllä `"TOUR_SECTION\n1\n3\n2\n-1\nEOF"` palautetaan `[0, 2, 1]`. Virheelliset formaatit (esim. puuttuva `-1`) palauttavat `None`.
*   **`read_tsp_file()`:** Testattu `.tsp`-tiedostojen lukeminen. Esimerkiksi sisällöllä `"EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0 0\n2 10 0\nEOF"` palautetaan `np.array([[0,0],[10,0]])`. Virheelliset formaatit (esim. ei-numeeriset koordinaatit) aiheuttavat virheen tai palauttavat tyhjän taulukon.

**C. Lin-Kernighan-algoritmin ydinkomponentit (`test_lk_core.py`)**

Algoritmin sisäisiä ydinosia on testattu `simple_tsp_setup`-fixturea (5 solmun ongelma) hyödyntäen.
*   **`step()`-funktio:** Testattu, että funktio löytää yksinkertaisen 2-opt-parannuksen. Esimerkiksi, jos kierros on `[0,1,2,3,4]` ja sen kustannus on 10, ja on olemassa 2-opt-siirto, joka rikkoo reunat `(0,1)` ja `(2,3)` ja muodostaa reunat `(0,2)` ja `(1,3)`, ja tämä siirto laskee kustannuksen 8:aan, `step()`-funktion odotetaan löytävän tämän ja palauttavan vastaavan siirtosarjan.
*   **`alternate_step()`-funktio:** Testattu vastaavasti, että funktio tunnistaa tietyt 3-opt- tai 5-opt-parannukset, jos sellaisia on saatavilla.
*   **`lk_search()`-funktio:** Testattu, että yksi `lk_search`-kierros pystyy parantamaan ei-optimaalisen kierroksen tunnettuun optimiin `simple_tsp_setup`-instanssissa.
*   **`lin_kernighan()`-pääalgoritmi:** Testattu, että algoritmi parantaa ei-optimaalisen aloituskierroksen tunnettuun optimiin `simple_tsp_setup`-instanssissa (kustannus `8.00056...`). Testattu myös, että se ei tee muutoksia, jos aloitetaan jo optimaalisella kierroksella.

**D. Lin-Kernighan-algoritmit (`test_lk_algorithms.py`)**

*   **`lin_kernighan()` ja `chained_lin_kernighan()`:** Testattu konvergoituminen tunnettuun optimiin `simple_tsp_setup`-instanssissa. Testattu myös aikarajojen noudattaminen ja toiminta, kun aloitetaan optimaalisella kierroksella.
*   **Pysähtyminen `known_optimal_length` saavutettaessa:** Testattu `rand4`-instanssilla (4 solmua, luetaan tiedostoista). Kun `chained_lin_kernighan` ajetaan tälle instanssille ja annetaan sen tunnettu optimipituus, algoritmin odotetaan pysähtyvän nopeasti löydettyään tämän pituuden, vaikka aikaraja olisi pidempi.

## 3. Minkälaisilla syötteillä testaus tehtiin?

Testauksessa käytettiin monipuolisia syötteitä:
*   **Manuaalisesti määritellyt pienet instanssit:** Useita testejä varten luotiin pieniä, käsin laskettavissa olevia esimerkkejä solmujen koordinaateista ja kierroksista (esim. 3-5 solmua).
*   **`simple_tsp_setup`-fixture:** Tämä `conftest.py`-tiedostossa määritelty fixture tarjoaa konsistentin 5 solmun TSP-instanssin, johon kuuluu koordinaatit, etäisyysmatriisi, alkukierros, Delaunay-naapurit ja tunnettu optimaalinen kierros (`[0, 2, 4, 3, 1]`, kustannus n. `8.0006`). Tätä käytetään laajasti LK-ydinkomponenttien ja -algoritmien testauksessa.
*   **Reunatapaukset:** Tyhjät kierrokset, yhden tai kahden solmun kierrokset.
*   **Tiedostopohjaiset syötteet:** `read_opt_tour` ja `read_tsp_file` -funktioiden testauksessa luotiin dynaamisesti testitiedostoja (`tmp_path`-fixture) simuloimaan validia ja virheellistä TSPLIB-formaatin dataa.
*   **Ulkoiset TSPLIB-tiedostot:** Yksi testi (`test_chained_lk_terminates_at_known_optimum`) käyttää `rand4.tsp`- ja `rand4.opt.tour`-tiedostoja `verifications/random`-kansiosta.
*   **Algoritmin parametrit:** `LK_CONFIG`-sanakirjan eri asetuksia (esim. `MAX_LEVEL`, `BREADTH`, `TIME_LIMIT`) muunneltiin testaamaan algoritmien käyttäytymistä eri rajoitteilla.

## 4. Miten testit voidaan toistaa?

Testit on kirjoitettu `pytest`-testauskehyksellä ja ne voidaan toistaa seuraavasti:
1.  **Asenna riippuvuudet:** Varmista, että Python-ympäristöön on asennettu `pytest`, `numpy` ja `scipy`.
2.  **Siirry projektin juurihakemistoon.**
3.  **Suorita `pytest`:** Aja komento `pytest` terminaalissa.
    *   Testit käyttävät pääosin dynaamisesti luotuja syötteitä tai `simple_tsp_setup`-fixturea, jotka eivät vaadi ulkoisia tiedostoja (lukuun ottamatta `rand4`-testiä, joka vaatii `verifications/random/rand4.tsp` ja `verifications/random/rand4.opt.tour`).
    *   Satunnaisuutta käytetään `double_bridge`-funktiossa, mutta testit on suunniteltu varmistamaan sen yleiset ominaisuudet. Yhdessä `double_bridge`-testissä käytetään `np.random.seed(42)` toistettavuuden varmistamiseksi.

## 5. Ohjelman toiminnan mahdollisen empiirisen testauksen tulosten esittäminen graafisessa muodossa

Toimitetut yksikkötestit keskittyvät ohjelman eri osien oikeellisuuden varmistamiseen. Itse pääratkaisijaskripti (`lin_kernighan_tsp_solver.py`) sisältää toiminnallisuuden (`plot_all_tours` ja `display_summary_table`), joka visualisoi ja taulukoi tulokset useille TSPLIB-instansseille. Tämä toimii empiirisenä tulosten esityksenä algoritmin suorituskyvystä.

Mahdollisia jatkokehitysideoita empiiriseen testaukseen voisivat olla skaalautuvuustestit (ajoaika vs. solmumäärä) tai `LK_CONFIG`-parametrien vaikutuksen systemaattinen tutkiminen, joiden tulokset esitettäisiin graafeina. Nämä ovat kuitenkin erillisiä yksikkötestauksesta.