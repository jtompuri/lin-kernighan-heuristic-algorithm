# Testausdokumentti

Tämä dokumentti kuvaa Lin-Kernighan TSP-ratkaisijan Python-toteutuksen testausprosessia. Testaus on suoritettu kattavilla automaattisilla yksikkötesteillä ja integraatiotesteillä `pytest`-kehystä hyödyntäen.

## 1. Yksikkötestauksen kattavuusraportti

Yksikkötestien kattavuus on tällä hetkellä 99%. Yksityiskohtainen kattavuusraportti löytyy html-muodossa kansiosta [`htmlcov`](/htmlcov/index.html). Et voi selata raporttia Githubissa, vaan sinun pitää kloonata projekti paikalliseen kansioon ja avata index.html-sivu selaimeen. Yksikkötestauksen kattavuusraportti on tuotettu komennolla `pytest --cov=lin_kernighan_tsp_solver --cov-report=html --cov-report=term --cov-report=term-missing`.

```bash
Name                                           Stmts   Miss  Cover   Missing
----------------------------------------------------------------------------
lin_kernighan_tsp_solver/__init__.py               0      0   100%
lin_kernighan_tsp_solver/__main__.py               2      0   100%
lin_kernighan_tsp_solver/config.py                13      0   100%
lin_kernighan_tsp_solver/lk_algorithm.py         352      0   100%
lin_kernighan_tsp_solver/main.py                 133      3    98%   83, 154, 156
lin_kernighan_tsp_solver/starting_cycles.py      125      3    98%   71, 188, 190
lin_kernighan_tsp_solver/tsp_io.py                82      0   100%
lin_kernighan_tsp_solver/utils.py                 89      1    99%   91
----------------------------------------------------------------------------
TOTAL                                             796      7    99%
```

## 2. Mitä on testattu, miten tämä tehtiin?

Ohjelman eri komponentteja on testattu kattavasti seuraavasti:

**A. `Tour`-luokka (`test_tour.py`)**

`Tour`-luokka, joka edustaa kierrosta ja sen operaatioita, on testattu kattavasti.
*   **Alustus ja `get_tour()`:** Testattu, että `Tour`-olio alustuu oikein annetulla solmujärjestyksellä ja että `get_tour()`-metodi palauttaa tämän järjestyksen, normalisoiden sen alkamaan solmusta 0, jos se on kierroksessa mukana (esim. `test_init_and_get_tour`, `test_tour_get_tour_normalization`).
*   **`next()` ja `prev()`:** Varmistettu, että metodit palauttavat oikean seuraajan ja edeltäjän kierroksella eri solmuille (esim. `test_next_prev`, `test_tour_next_prev_parametrized`). Testattu myös tyhjän kierroksen (`IndexError` testissä `test_tour_next_prev_on_empty_tour`) ja virheellisten syötteiden (solmu ei kierroksessa tai `pos`-taulukon ulkopuolella, `IndexError` testissä `test_tour_next_prev_node_not_in_tour_if_pos_small`) käsittely.
*   **`sequence()`:** Testattu, tunnistaako metodi oikein, onko tietty solmu annetulla kierroksen segmentillä (palauttaa `True`/`False`) (esim. `test_sequence_wrap_and_nonwrap`, `test_tour_sequence`). Myös tyhjän kierroksen käsittely on testattu (`test_tour_empty_sequence`, palauttaa `False`).
*   **`flip()`:** Testattu segmentin kääntämisen oikeellisuus sekä ylittävillä että ei-ylittävillä segmenteillä (esim. `test_flip_no_wrap`, `test_flip_wrap`, `test_tour_flip_scenarios`). Varmistettu, että sisäinen `order`-taulukko ja `pos`-taulukko päivittyvät oikein.
*   **Kustannusten laskenta (`__init__` ja `init_cost`):** Testattu, että kierroksen kustannus lasketaan oikein (esim. `test_tour_init_with_cost_specific_sequence`, `test_tour_init_cost_parametrized`). Myös tyhjien (`test_tour_init_cost_empty`) ja yhden solmun kierrosten kustannukset on varmistettu.
*   **`flip_and_update_cost()`:** Testattu, että metodi kääntää segmentin oikein ja päivittää kierroksen kokonaiskustannuksen oikein laskien muutoskustannuksen (delta) (esim. `test_tour_flip_and_update_cost_basic`, `test_tour_flip_and_update_cost_parametrized`). Testattu myös tyhjän kierroksen tapaus (`test_tour_empty_flip_and_update_cost`, delta 0.0). Myös tilanne, jossa kierroksen kustannus on aluksi `None` ja lasketaan uudelleen, on katettu (`test_flip_and_update_cost_recomputes_cost_if_none` `test_lk_algorithm.py`:ssä).

**B. Aputoiminnot (`test_utils.py`)**

Moduulien `lk_algorithm.py`, `tsp_io.py`, ja `utils.py` keskeiset aputoiminnot on testattu `test_utils.py`-tiedostossa.
*   **`build_distance_matrix()` (moduulista `lk_algorithm.py`):** Testattu etäisyysmatriisin laskenta (esim. `test_build_distance_matrix_simple_cases`, `test_build_distance_matrix`). Myös tyhjä (`test_build_distance_matrix_edge_cases`) ja yhden solmun tapaukset (`test_build_distance_matrix_single_node`) on testattu kattavasti.
*   **`delaunay_neighbors()` (moduulista `lk_algorithm.py`):** Testattu Delaunay-naapurien löytäminen eri pistemäärillä (0, 1, 2, 3, 4 pistettä) ja `simple_tsp_setup`-fixturen koordinaateilla (esim. `test_delaunay_neighbors_few_points`, `test_delaunay_neighbors_triangle`, `test_delaunay_neighbors_from_fixture`). Varmistettu naapurilistojen perusominaisuudet.
*   **`double_bridge()` (moduulista `lk_algorithm.py`):** Testattu "double bridge" -potkutoimintoa (esim. `test_double_bridge_kick`, `test_kick_function_perturbs_tour`). Varmistettu, että se tuottaa validin permutaation, muuttaa kierrosta (solmuja > 4) ja palauttaa alkuperäisen (solmuja <= 4).
*   **`read_opt_tour()` (moduulista `tsp_io.py`):** Testattu `.opt.tour`-tiedostojen lukeminen (esim. `test_read_opt_tour`). Testattu useita virheellisiä formaatteja (`test_read_opt_tour_error_cases`) ja yleinen tiedostonlukuvirhe (`test_read_opt_tour_general_exception`).
*   **`read_tsp_file()` (moduulista `tsp_io.py`):** Testattu `.tsp`-tiedostojen lukeminen (vain `EUC_2D`) (esim. `test_read_tsp_file_valid_euc_2d`). Testattu virheellisiä formaatteja (`test_read_tsp_file_error_cases`, esim. ei-tuettu `EDGE_WEIGHT_TYPE`, virheelliset koordinaattirivit) ja `FileNotFoundError` (`test_read_tsp_file_not_found`).
*   **`write_opt_tour()` (moduulista `tsp_io.py`):** Testattu kierroksen tallentaminen optimaalisena ratkaisuna `.opt.tour`-formaatissa (esim. `test_write_opt_tour_valid_tour`, `test_write_opt_tour_with_comments`).
*   **`display_summary_table()` ja `plot_all_tours()` (moduulista `utils.py`):** Testattu tulosten esittäminen taulukko- ja kuvamuodossa, mukaan lukien reunatapaukset ja virhetilanteet.

**C. Lin-Kernighan-algoritmin ydinkomponentit (`test_lk_algorithm.py`)**

Algoritmin sisäisiä ydinosia on testattu pääasiassa `simple_tsp_setup`-fixturea hyödyntäen.
*   **`step()`-funktio:** Testattu 2-opt-parannuksen löytäminen (`test_step_finds_simple_2_opt`) ja toiminta optimaalisella kierroksella (`test_step_no_improvement_on_optimal_tour`). Aikarajojen käsittely on testattu (`test_step_deadline_in_standard_flips_loop`). `BREADTH`-rajoitteen vaikutus (`test_step_hits_break_due_to_breadth_limit_zero`) ja tilanteet, joissa kandidaatteja ei löydy (`test_step_no_candidates_for_t1_t2_pair`, `test_step_returns_false_none_when_no_candidates`), on katettu. Myös virheellisten kandidaattien ohittaminen on testattu (`test_step_skips_invalid_y1_candidates`).
*   **`alternate_step()`-funktio:** Testattu parannusten löytäminen (`test_alternate_step_finds_improvement`) ja toiminta optimaalisella kierroksella (`test_alternate_step_no_improvement_on_optimal`). Testattu aikarajan ylittymisen käsittely yleisesti ja tarkemmin eri silmukoiden alussa. `BREADTH_A/B/D`-rajoitteiden vaikutus ja kandidaattien ohittaminen, jos ne ovat `t1`, `t2` tai `chosen_y1`, on testattu. Myös tilanteet, joissa kandidaatteja ei löydy, on katettu.
*   **`lk_search()`-funktio:** Testattu parannuksen löytäminen ja toiminta optimaalisella kierroksella. Aikarajojen käsittely: välitön paluu, jos aikaraja ylittynyt alussa, ja paluu, jos aikaraja ylittyy `step()`-kutsun jälkeen mutta ennen `alternate_step()`-kutsua. Ei-parantavan `alternate_step`-tuloksen käsittely ja tilanne, jossa parannusta ei löydy, on testattu.
*   **`lin_kernighan()`-pääalgoritmi:** Testattu konvergoituminen optimiin ja toiminta optimaalisella aloituskierroksella. Testattu pienillä instansseilla (2 ja 3 solmua). Aikarajojen noudattaminen ja `LK_CONFIG`-parametrien vaikutus on testattu.

**D. Ketjutettu Lin-Kernighan -algoritmi (`test_chained_lk.py`)**

Iteratiivista `chained_lin_kernighan`-algoritmia on testattu.
*   **Konvergoituminen ja optimaalisuus:** Testattu konvergoituminen tunnettuun optimiin ja aloitettaessa optimaalisella kierroksella.
*   **Pysähtyminen `known_optimal_length` saavutettaessa:** Testattu useilla `rand`N-instansseilla, että algoritmi pysähtyy löydettyään annetun optimipituuden. Myös tarkemmat testit pysähtymisestä optimiin yhden tai useamman `lin_kernighan`-kutsun jälkeen.
*   **Aikarajojen noudattaminen:** Testattu, että yleinen aikaraja pysäyttää algoritmin.
*   **Iteraatioiden hallinta:** Testattu toiminta, kun `MAX_NO_IMPROVEMENT_ITERATIONS` saavutetaan. Yhden iteraation tapaus ja useamman iteraation toiminta on varmistettu.
*   **Reunatapaukset:** Testattu toiminta tyhjällä aloituskierroksella.

**E. Aloituskierrosten algoritmeissa (`test_starting_cycles.py`)**

Erilaisten aloituskierrosten generointialgoritmeja on testattu kattavasti.
*   **`random_cycle()`:** Testattu satunnaisten kierrosten generointi eri solmumäärillä, mukaan lukien reunatapaukset (0-2 solmua).
*   **`greedy_cycle()`:** Testattu ahne algoritmi, joka valitsee aina lähimmän käymättömän solmun. Testattu eri konfiguraatioilla ja solmumäärillä.
*   **`savings_cycle()`:** Testattu Clarke-Wright Savings -algoritmi, joka optimoi reittien yhdistämistä säästöjen perusteella. Varmistettu kierrosten validius ja kustannuslaskenta.
*   **`nearest_neighbor_cycle()`:** Testattu lähimmän naapurin algoritmi eri aloitussolmuilla. Testattu deterministisyys ja tulosten oikeellisuus.
*   **`kruskal_cycle()`:** Testattu Kruskalin algoritmin muunnos TSP-kierrosten luomiseen. Varmistettu, että tuloksena on validi Hamiltonin kierros.
*   **`prim_cycle()`:** Testattu Primin algoritmin muunnos TSP-kierrosten luomiseen. Testattu eri aloitussolmuilla ja varmistettu kierrosten validius.
*   **`convex_hull_cycle()`:** Testattu konveksiin kuoreen perustuva aloituskierros. Testattu eri geometrisilla konfiguraatioilla, mukaan lukien kollineaariset pisteet.
*   **`two_opt_on_random()`:** Testattu 2-opt-optimointi satunnaiselle kierrokselle aloituskierrokseksi. Varmistettu, että tulos on parempi tai yhtä hyvä kuin alkuperäinen.

**F. Komentorivikäyttöliittymä (`test_main.py`, `test_main_helpers.py`, `test_cli_files.py`)**

Sovelluksen CLI-toiminnallisuutta on testattu kattavasti:
*   **Argumenttien käsittely:** Testattu eri komentoriviparametrit, mukaan lukien tiedostopolut, aikarajat, aloitusalgoritmit ja tallennusasetukset.
*   **Tiedostojen käsittely:** Testattu TSP- ja OPT-tiedostojen lukeminen ja kirjoittaminen eri formaateissa.
*   **Virheenkäsittely:** Testattu virhetilanteiden käsittely, kuten puuttuvat tiedostot, virheelliset parametrit ja I/O-virheet.
*   **Tulosten tallennus:** Testattu kierrosten tallentaminen eri formaateissa (`test_tour_saving.py`).

**G. Aloitusalgoritmien CLI-testaus (`test_starting_cycles_cli.py`)**

Aloitusalgoritmien komentorivikäyttöä on testattu:
*   **Kaikkien algoritmien testaus:** Varmistettu, että jokainen aloitusalgoritmi toimii CLI:n kautta oikein.
*   **Parametrien välitys:** Testattu, että CLI-parametrit välitetään oikein algoritmeille.
*   **Tulosten validointi:** Varmistettu, että eri aloitusalgoritmit tuottavat valideja kierroksia.

**H. Työnkulun hallinta (`test_workflow.py`)**

Pääohjelman (`main.py`) toimintoja on testattu.
*   **`process_single_instance`:** Testattu virhetilanteet (koordinaattien lataus, TSP-tiedoston lukuvirhe, optimaalista kierrosta ei löydy, algoritmin sisäinen virhe) ja `gap`-arvon laskenta.
*   **`main` (pääfunktio):** Testattu virhetilanteet (TSP-kansion puuttuminen, ei TSP-tiedostoja) ja onnistuneet käsittelyt. Varmistettu yhteenvetotaulukon ja kuvaajien kutsuminen.

**I. Tulosten esitys (`test_output.py`)**

Moduulin `utils.py` tulosten visualisointi- ja yhteenvetofunktioita on testattu.
*   **`display_summary_table()`:** Testattu useilla reunatapauksilla, kuten tyhjällä tulosdatalla tai kun `opt_len` tai `gap` puuttuvat.
*   **`plot_all_tours()`:** Testattu toiminta tyhjällä tulosdatalla, varoitusten tulostuminen ja onnistunut piirtopolku mockaamalla `matplotlib`-kutsut.

## 3. Minkälaisilla syötteillä testaus tehtiin?

Testauksessa käytettiin monipuolisia syötteitä:
*   **Manuaalisesti määritellyt pienet instanssit:** Useita testejä varten luotiin pieniä, käsin laskettavissa olevia esimerkkejä solmujen koordinaateista ja kierroksista (esim. 2-6 solmua).
*   **`simple_tsp_setup`-fixture:** Tämä `conftest.py`-tiedostossa määritelty fixture tarjoaa konsistentin 5 solmun TSP-instanssin (koordinaatit `[[0.0, 0.01], [1.0, -0.01], [3.0, 0.02], [2.0, -0.02], [4.0, 0.0]]`), johon kuuluu etäisyysmatriisi, alkukierros (`[0, 1, 2, 3, 4]`), Delaunay-naapurit ja tunnettu optimaalinen kierros (`[0, 2, 4, 3, 1]`, kustannus n. `8.000567`). Tätä käytetään laajasti LK-ydinkomponenttien ja -algoritmien testauksessa.
*   **Reunatapaukset:** Tyhjät kierrokset, yhden tai kahden solmun kierrokset, virheelliset syötteet funktioille, tyhjät tiedostot.
*   **Tiedostopohjaiset syötteet:** `read_opt_tour` ja `read_tsp_file` -funktioiden testauksessa luotiin dynaamisesti testitiedostoja (`tmp_path`-fixture) simuloimaan validia ja virheellistä TSPLIB-formaatin dataa.
*   **Ulkoiset TSPLIB-tiedostot:** Useita testejä käyttää parametrisoidusti `rand4.tsp` ... `rand12.tsp` ja vastaavia `.opt.tour`-tiedostoja `problems/random`-kansiosta sekä muita TSPLIB-instansseja `problems/tsplib95`-kansiosta.
*   **Aloitusalgoritmien testaus:** Jokaista aloitusalgoritmia testattiin eri kokoisilla ja geometrisesti erilaisilla instansseilla, mukaan lukien reunatapaukset kuten kollineaariset pisteet konveksille kuorelle.
*   **CLI-parametrit:** Komentorivikäyttöliittymän testaus kattoi kaikki mahdolliset parametrikombinaatiot, virheelliset syötteet ja tiedostopolut.
*   **Algoritmin parametrit:** `LK_CONFIG`-sanakirjan eri asetuksia (esim. `MAX_LEVEL`, `BREADTH`, `TIME_LIMIT`, `MAX_NO_IMPROVEMENT_ITERATIONS`) muunneltiin testaamaan algoritmien käyttäytymistä eri rajoitteilla.

## 4. Miten testit voidaan toistaa?

Testit on kirjoitettu `pytest`-testauskehyksellä ja ne voidaan toistaa seuraavasti:

1.  **Asenna riippuvuudet:** Varmista, että Python-ympäristöön on asennettu kaikki tarvittavat paketit:
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```
    Tämä asentaa `pytest`, `pytest-cov`, `numpy`, `scipy`, `matplotlib` ja muut tarvittavat kirjastot.

2.  **Siirry projektin juurihakemistoon:** Varmista, että olet hakemistossa, jossa `pytest.ini`-tiedosto sijaitsee.

3.  **Suorita kaikki testit:**
    ```bash
    pytest
    ```

4.  **Suorita testit kattavuusraportilla:**
    ```bash
    pytest --cov=lin_kernighan_tsp_solver --cov-report=html --cov-report=term
    ```
    Tämä luo HTML-kattavuusraportin `htmlcov/`-kansioon.

5.  **Suorita tietty testitiedosto:**
    ```bash
    pytest tests/test_tour.py
    pytest tests/test_starting_cycles.py
    ```

6.  **Suorita tietty testi:**
    ```bash
    pytest tests/test_tour.py::test_tour_init_and_get_tour
    ```

**Huomioita toistettavuudesta:**
*   Testit käyttävät pääosin deterministisiä syötteitä tai `simple_tsp_setup`-fixturea.
*   Satunnaisuutta käyttävät testit (esim. `double_bridge`, `random_cycle`) on suunniteltu testaamaan yleisiä ominaisuuksia.
*   Ulkoiset TSPLIB-tiedostot ovat mukana repositoriossa `problems/`-kansiossa.
*   Testit eivät vaadi internetyhteyttä tai ulkoisia palveluita.
*   Mockauksia käytetään `matplotlib`-visualisoinneissa välttämään riippuvuudet graafisista järjestelmistä.

**Testien suoritus CI/CD-ympäristössä:**
Testit on suunniteltu toimimaan automatisoiduissa ympäristöissä ilman käyttäjän vuorovaikutusta.

## 5. Ohjelman toiminnan empiirisen testauksen tulokset graafisessa muodossa

Toimitetut yksikkötestit keskittyvät ohjelman eri osien oikeellisuuden varmistamiseen. Pääskripti `lin_kernighan_tsp_solver.py` sisältää toiminnallisuuden, joka visualisoi ja taulukoi tulokset useille TSPLIB-instansseille. Tämä toimii empiirisenä tulosten esityksenä algoritmin suorituskyvystä. Seuraavassa esitettävät kuvaajat on luotu 20 sekunnin aikarajalla ellei toisin ole mainittu.

### LK-algoritmin verfiointi brute force -algoritmin tuloksilla

![LK verifiointi random 20s](/images/lk_verifications_random_20s.png)

```
Configuration parameters:
  MAX_LEVEL   = 12
  BREADTH     = [5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  BREADTH_A   = 5
  BREADTH_B   = 5
  BREADTH_D   = 1
  TIME_LIMIT  = 20.00

Instance     OptLen   HeuLen   Gap(%)  Time(s)
----------------------------------------------
rand10      2862.79  2862.79     0.00     0.01
rand11      2866.43  2866.43     0.00     0.01
rand12      2894.02  2894.02     0.00     0.01
rand4       2032.77  2032.77     0.00     0.00
rand5       2079.70  2079.70     0.00     0.00
rand6       2127.74  2127.74     0.00     0.00
rand7       2147.16  2147.16     0.00     0.00
rand8       2149.50  2149.50     0.00     0.00
rand9       2829.56  2829.56     0.00     0.00
----------------------------------------------
SUMMARY    21989.68 21989.68     0.00     0.00
```

Skriptillä `create_tsp_problem.py` luotiin 4–12 solmun satunnaisia tsp-ongelmia, joiden optimaalinen reitti ratkaistiin skriptillä `exact_tsp_solver.py` käymällä läpi kaikki mahdolliset reitit. LK-algoritmin oikeellisuus pienillä solmujen määrällä varmistettiin ratkaisemalla tsp-ongelmat ja vertaamalla tulosta optimaaliseen reittiin. Kuten kuvasta näkyy, niin LK-algoritmi löysi optimaalisen ratkaisun kaikkiin tsp-ongelmiin.  

### LK-algoritmin verfiointi TSPLIB95-datalla

![TSPLIB95-kuvaajat](/images/lk_verifications_tsplib95_20s.png)

```
Configuration parameters:
  MAX_LEVEL   = 12
  BREADTH     = [5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  BREADTH_A   = 5
  BREADTH_B   = 5
  BREADTH_D   = 1
  TIME_LIMIT  = 20.00

Instance     OptLen   HeuLen   Gap(%)  Time(s)
----------------------------------------------
a280        2586.77  2617.49     1.19    20.00
berlin52    7544.37  7544.37     0.00     0.19
ch130       6110.86  6164.24     0.87    20.00
ch150       6532.28  6552.30     0.31    20.00
eil101       642.31   640.21     0.00    20.00
eil51        429.98   428.87     0.00    20.00
eil76        545.39   544.37     0.00    20.00
kroA100    21285.44 21285.44     0.00    11.69
kroC100    20750.76 20750.76     0.00     2.38
kroD100    21294.29 21294.29     0.00    11.22
lin105     14383.00 14383.00     0.00     2.70
pcb442     50783.55 74322.54    46.35    20.00
pr1002     259066.66 339278.81    30.96    20.00
pr2392     378062.83 378062.83     0.00    12.97
pr76       108159.44 108159.44     0.00     2.04
rd100       7910.40  7910.40     0.00     2.35
st70         678.60   677.11     0.00    20.00
tsp225      3859.00  3936.55     2.01    20.00
----------------------------------------------
SUMMARY    910625.92 1014553.02     4.54    13.64
```

TSPLIB95-kirjastosta poimittiin kaikki tsp-ongelmat, joihin oli tarjolla optimaalinen ratkaisu. Tämän jälkeen varmistettiin LK-algoritmin oikeellisuus ratkaisemalla tsp-ongelmat ja vertaamalla saatua ratkaisua optimaaliseen ratkaisuun. Vertailuluku `gap` kertoo kuinka monta prosenttia löydetty reitti on optimaalista reittiä pidempi. Kuten nähdään niin löydetyt reitit ovat kahta poikkeusta lukuunottamatta 0-2% etäisyydellä optimaalisesta ratkaisusta eli erittäin lähellä. 


### LK-algoritmin tulokset pidemmällä aikarajalla

![LK-algoritmi pitkällä aikarajalla](/images/lk_verifications_tsplib95_900s.png)

```
Configuration parameters:
  MAX_LEVEL   = 12
  BREADTH     = [5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  BREADTH_A   = 5
  BREADTH_B   = 5
  BREADTH_D   = 1
  TIME_LIMIT  = 900.00s

Instance   OptLen   HeuLen   Gap(%)   Time(s)
---------------------------------------------
a280        2586.77  2586.77     0.00   394.77
berlin52    7544.37  7544.37     0.00     0.17
ch130       6110.86  6110.72     0.00  1779.75
ch150       6532.28  6530.90     0.00  5213.94
eil101       642.31   642.03     0.00  1501.20
eil51        429.98   428.98     0.00  2248.01
eil76        545.39   544.37     0.00  2429.44
kroA100    21285.44 21285.44     0.00     2.17
kroC100    20750.76 20750.76     0.00     4.94
kroD100    21294.29 21375.45     0.38  1998.50
lin105     14383.00 14383.00     0.00     2.53
pcb442     50783.55 78633.19    54.84  2964.19
pr1002     259066.66 342394.27    32.16  2642.24
pr2392     378062.83 378062.83     0.00  1691.25
pr76       108159.44 108159.44     0.00     0.98
rd100       7910.40  7910.40     0.00     3.88
st70         678.60   677.11     0.00   900.00
tsp225      3859.00  3859.00     0.00    62.48
---------------------------------------------
SUMMARY    910625.92 1021879.03     4.85  1324.47
```

Havaitaan, että pidemmällä aikarajalla lähestytään optimaalista ratkaisua kahta poikkeusta lukuunottamatta. Otetaan toinen poikkeuksista lähempään tarkasteluun, jotta saadaan selville, jääkö LK-algoritmi paikalliseen minimiin vai onko optimaalisen ratkaisun löytäminen vain hidasta.

### Poikkeustapauksen `pcb422` lähempi tarkastelu 

![pcb422-ongelma](/images/lk_verification_pcb442_7200s.png)

```
Configuration parameters:
  MAX_LEVEL   = 12
  BREADTH     = [5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  BREADTH_A   = 5
  BREADTH_B   = 5
  BREADTH_D   = 1
  TIME_LIMIT  = 7200.00

Instance     OptLen   HeuLen   Gap(%)  Time(s)
----------------------------------------------
pcb442     50783.55 51370.13     1.16  7389.95
----------------------------------------------
SUMMARY    50783.55 51370.13     1.16  7389.95
```

Tutkittiin lähemmin tsp-ongelmaa `pcb422` asettamalla aikaraja 7200 sekuntiin eli kahteen tuntiin. Pitkällä aikarajalla myös tämän tsp-ongelman likiarvo saatiin lähelle optimaalista ratkaisua (1.16%). Toisin sanoen LK-algoritmi ei jää jumiin paikalliseen minimiin. 

### Yksinkertainen tsp-ratkaisija (pienet tsp-ongelmat)

![Simple tsp-solver 1](/images/simple_tsp_solver_verifications_random_20s.png)

```
--------------------------------------------------
Instance     OptLen   HeuLen   Gap(%)  Time(s)
--------------------------------------------------
rand10      2862.79  2914.22     1.80     0.00
rand11      2866.43  2947.14     2.82     0.00
rand12      2894.02  2894.02     0.00     0.00
rand4       2032.77  2032.77     0.00     0.00
rand5       2079.70  2145.25     3.15     0.00
rand6       2127.74  2145.26     0.82     0.00
rand7       2147.16  2227.92     3.76     0.00
rand8       2149.50  2149.50     0.00     0.00
rand9       2829.56  2910.33     2.85     0.00
--------------------------------------------------
SUMMARY    21989.68 22366.42     1.69     0.00
```

Yksinkertainen tsp-ratkaisija `simple_tsp_solver.py` toteuttaa rekursiivisesti 2-opt-vaihtoja, mutta ei sisällä mitään kehittyneempiä LK-algoritmin ominaisuuksia, kuten `double_bridge` tai `chained_lk_search`. Havaitaan, että yksinkertainen algoritmi ei saa ratkaistua optimaalisesti kaikkia 4-12 solmun tsp-ongelmia.

### Yksinkertainen tsp-ratkaisija (tsplib95-ongelmat)


![Simple tsp-solver 2](/images/simple_tsp_solver_verifications_tsplib95_20s.png)

```
--------------------------------------------------
Instance     OptLen   HeuLen   Gap(%)  Time(s)
--------------------------------------------------
a280        2586.77  2727.23     5.43     0.54
berlin52    7544.37  8002.77     6.08     0.05
ch130       6110.86  6470.60     5.89     1.32
ch150       6532.28  7103.04     8.74     2.30
eil101       642.31   678.19     5.59     0.71
eil51        429.98   443.60     3.17     0.09
eil76        545.39   572.96     5.06     0.25
kroA100    21285.44 21605.87     1.51     0.75
kroC100    20750.76 21233.71     2.33     0.70
kroD100    21294.29 23317.94     9.50     0.75
lin105     14383.00 15549.63     8.11     0.30
pcb442     50783.55 91965.09    81.09     5.00
pr1002     259066.66 315885.78    21.93     5.00
pr2392     378062.83 378062.83     0.00     5.00
pr76       108159.44 118342.93     9.42     0.07
rd100       7910.40  8439.96     6.69     0.72
st70         678.60   743.87     9.62     0.23
tsp225      3859.00  4090.02     5.99     2.91
--------------------------------------------------
SUMMARY    910625.92 1025236.02    10.90     1.48
```

Havaitaan, että yksinkertainen tsp-ratkaisija pääsee yleensä noin 5-10 %:n päähän optimaalisesta ratkaisusta poislukien muutama poikkeustapaus. Voidaan todeta, että LK-algoritmi tuottaa merkittävästi parempia tuloksia. Toisaalta jos sovellusalueella riittää karkea likiarvo tai jos nopeus on kriittistä, niin yksinkertainen tsp-ratkaisija voi olla riittävän hyvä.

## 6. Sovelluksen puutteet ja kehitysmahdollisuudet

Vaikka sovellus on testattu kattavasti ja saavuttaa 99% lausekattavuuden yksikkötesteissä, on olemassa joitakin tunnistettuja puutteita ja mahdollisia kehityskohteita:

**Testaamatta jääneet rivit (1% kattavuudesta):**
*   **`main.py` (3 riviä):** Kaksi `sys.exit()`-kutsua virhetilanteissa ja yksi tiedostojen listauksen virheenkäsittely. Nämä ovat vaikeasti testattavia ilman monimutkaisia mock-järjestelyjä.
*   **`starting_cycles.py` (3 riviä):** Kaksi `sys.exit()`-kutsua CLI-virhetilanteissa ja yksi virheenkäsittely. Vastaava tilanne kuin `main.py`-tiedostossa.
*   **`utils.py` (1 rivi):** Matplotlib-kuvaajan tallennuksen virheenkäsittely, joka on vaikea simuloida testissä.

**Teknisiä kehityskohteita:**
*   **TSPLIB-formaatin tuki:** Tällä hetkellä `read_tsp_file` tukee vain `EUC_2D`-tyyppisiä etäisyysmatriiseja. Laajempi tuki muille TSPLIB-formaateille (esim. `GEO`, `ATT`, eksplisiittiset matriisit) parantaisi sovelluksen käytettävyyttä.
*   **Naapurilistojen generointi:** Vain Delaunay-triangulaatioon perustuva naapurilista on implementoitu. Muiden naapuristrategioiden (esim. k-lähimmät naapurit, quadrant-naapurit) lisääminen voisi parantaa ratkaisun laatua tietyissä instansseissa.
*   **Parametrien viritys:** `LK_CONFIG`-sanakirjan parametrit ovat globaaleja ja niiden optimaaliset arvot voivat vaihdella instanssityypin ja -koon mukaan. Kehittyneempi parametrien hallinta tai automaattinen viritys voisi olla hyödyllistä.
*   **Suorituskyky suurilla instansseilla:** Vaikka algoritmi on tehokas, Pythonin luontainen hitaus verrattuna käännettyihin kieliin voi rajoittaa suorituskykyä erittäin suurilla TSP-instansseilla (useita tuhansia solmuja). Kriittisten osien optimointi tai siirtäminen Cythoniin voisi olla kehityssuunta.

**Käytettävyys ja toiminnallisuus:**
*   **Graafinen käyttöliittymä:** Nykyinen sovellus on komentorivipohjainen. Graafinen käyttöliittymä tai web-käyttöliittymä voisi parantaa käytettävyyttä.
*   **Virheiden raportointi:** Vaikka virheitä käsitellään kattavasti, käyttäjälle annettava palaute virhetilanteissa voisi olla yksityiskohtaisempaa ja ohjaavampaa.
*   **Rinnakkaiskäsittely:** Nykyinen toteutus ei hyödynnä useita prosessoriytimjä. Parallelisoiminen voisi parantaa suorituskykyä suurissa instansseissa.
*   **Aikarajapositiivinen ongelma:** On havaittu, että kun tietokone jätetään yksin laskemaan TSP-ratkaisua, algoritmin suoritus ei aina pysähdy asetettuun aikarajaan. Ongelma voi liittyä tietokoneen energiansäästö- tai lepotilaan.

**Testauksen kehityskohteet:**
*   **Pitkäaikaiset suorituskykytestit:** Nykyiset testit keskittyvät oikeellisuuteen. Automaattisia suorituskykytestejä voisi lisätä regressioiden havaitsemiseksi.
*   **Stress-testaus:** Testaus erittäin suurilla instansseilla muistin ja ajan kulutuksen suhteen.
*   **Integraatiotestit:** Vaikka CLI-testejä on paljon, end-to-end-testejä koko työkululle voisi lisätä.

Nämä kohdat tarjoavat suuntaviivoja sovelluksen jatkokehitykselle ja laadun parantamiselle. Tämänhetkinen 99%:n testikattavuus ja kattava testisarja tarjoavat kuitenkin vahvan perustan jatkokehitykselle.