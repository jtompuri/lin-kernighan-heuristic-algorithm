# Testausdokumentti

Tämä dokumentti kuvaa Lin-Kernighan TSP-ratkaisijan Python-toteutuksen testausprosessia. Testaus on suoritettu pääasiassa automaattisilla yksikkötesteillä `pytest`-kehystä hyödyntäen.

## 1. Yksikkötestauksen kattavuusraportti

Yksikkötestien kattavuus on tällä hetkellä 92%. Yksityiskohtainen kattavuusraportti löytyy html-muodossa kansiosta [`htmlcov`](/htmlcov/index.html). Et voi selata raporttia Githubissa, vaan sinun pitää kloonata projekti paikalliseen kansioon ja avata index.html-sivu selaimeen. Yksikkötestauksen kattavuusraportti on tuotettu komennolla `pytest --cov=lin_kernighan_tsp_solver --cov-report=html --cov-report=term --cov-report=term-missing`.

## 2. Mitä on testattu, miten tämä tehtiin?

Ohjelman eri komponentteja on testattu seuraavasti:

**A. `Tour`-luokka (`test_tour.py`)**

`Tour`-luokka, joka edustaa kierrosta ja sen operaatioita, on testattu kattavasti.
*   **Alustus ja `get_tour()`:** Testattu, että `Tour`-olio alustuu oikein annetulla solmujärjestyksellä ja että `get_tour()`-metodi palauttaa tämän järjestyksen, normalisoiden sen alkamaan solmusta 0, jos se on kierroksessa mukana. Esimerkiksi syötteellä `[1, 0, 2]` `get_tour()` palauttaa `[0, 2, 1]`.
*   **`next()` ja `prev()`:** Varmistettu, että metodit palauttavat oikean seuraajan ja edeltäjän kierroksella eri solmuille. Esimerkiksi kierroksella `[0, 1, 2]`, `tour.next(2)` palauttaa `0` ja `tour.prev(0)` palauttaa `2`. Testattu myös tyhjän kierroksen (`IndexError`) ja virheellisten syötteiden (solmu ei kierroksessa tai `pos`-taulukon ulkopuolella, `IndexError`) käsittely.
*   **`sequence()`:** Testattu, tunnistaako metodi oikein, onko tietty solmu annetulla kierroksen segmentillä. Esimerkiksi kierroksella `[0, 1, 2, 3, 4]`, `tour.sequence(0, 2, 4)` palauttaa `True`. Testattu myös ylittävät segmentit (esim. `tour.sequence(3, 1, 0)` palauttaa `True` solmulle `1`) ja tilanteet, joissa aloitussolmu ei ole kierroksessa (palauttaa tyhjän listan, vaikka dokumentaatio viittaa `True`/`False`-paluuarvoon, todellinen paluuarvo on segmentin solmut tai tyhjä lista, ja testit tarkistavat tämän). Myös tyhjän kierroksen käsittely on testattu.
*   **`flip()`:** Testattu segmentin kääntämisen oikeellisuus sekä ylittävillä että ei-ylittävillä segmenteillä. Esimerkiksi kierroksella `[0, 1, 2, 3, 4]`, `tour.flip(1, 3)` muuttaa kierroksen muotoon `[0, 3, 2, 1, 4]`. Varmistettu, että sisäinen `order`-taulukko ja `pos`-taulukko päivittyvät oikein.
*   **Kustannusten laskenta (`__init__` ja `init_cost`):** Testattu, että kierroksen kustannus lasketaan oikein. Esimerkiksi kolmion solmuilla `(0,0), (3,0), (0,4)` ja kierroksella `[0,1,2]`, kustannus on `3 + 5 + 4 = 12`. Myös tyhjien ja yhden solmun kierrosten kustannukset (0.0 tai `None`) on varmistettu.
*   **`flip_and_update_cost()`:** Testattu, että metodi kääntää segmentin oikein ja päivittää kierroksen kokonaiskustannuksen oikein laskien muutoskustannuksen (delta). Esimerkiksi kierroksella `[0,1,2,3]` ja etäisyyksillä `D[0,1]=1, D[1,2]=1, D[2,3]=1, D[3,0]=1` (kustannus 4), `flip_and_update_cost(1,2)` (kääntää `[1,2]` -> `[2,1]`) muuttaa kierroksen `[0,2,1,3]`. Jos `D[0,2]=1.5, D[1,3]=1.5, D[0,1]=1, D[2,3]=1`, niin vanhat reunat `(0,1)` ja `(2,3)` (summa 2) korvautuvat reunoilla `(0,2)` ja `(1,3)` (summa 3). Kustannusmuutos on `+1`, ja uusi kustannus on `5`. Testattu myös tyhjän kierroksen tapaus.

**B. Aputoiminnot (`test_utils.py`)**

Moduulin keskeiset aputoiminnot on testattu.
*   **`build_distance_matrix()`:** Testattu etäisyysmatriisin laskenta. Esimerkiksi koordinaateilla `[[0,0], [3,0], [0,4]]`, matriisin tulee olla `[[0,3,4], [3,0,5], [4,5,0]]` (pyöristettynä). Myös tyhjä ja yhden solmun tapaukset on testattu kattavasti.
*   **`delaunay_neighbors()`:** Testattu Delaunay-naapurien löytäminen eri pistemäärillä (0, 1, 2, 3, 4 pistettä) ja `simple_tsp_setup`-fixturen koordinaateilla. Varmistettu naapurilistojen perusominaisuudet (järjestys, ei itsesilmukoita, symmetrisyys).
*   **`double_bridge()`:** Testattu "double bridge" -potkutoimintoa. Varmistettu, että se tuottaa validin permutaation alkuperäisestä kierroksesta, muuttaa kierrosta, jos solmuja on > 4, ja palauttaa alkuperäisen kierroksen, jos solmuja on <= 4. Toistettavuus on varmistettu `np.random.seed()`-kutsulla yhdessä testissä.
*   **`read_opt_tour()`:** Testattu `.opt.tour`-tiedostojen lukeminen. Esimerkiksi tiedostosisällöllä `"TOUR_SECTION\n1\n3\n2\n-1\nEOF"` palautetaan `[0, 2, 1]`. Testattu useita virheellisiä formaatteja (esim. puuttuva `-1`, puuttuva `TOUR_SECTION`, ei-numeeriset solmut, tyhjä tiedosto), jotka palauttavat `None`. Myös yleinen tiedostonlukuvirhe (`OSError`) on testattu.
*   **`read_tsp_file()`:** Testattu `.tsp`-tiedostojen lukeminen (vain `EUC_2D`). Esimerkiksi sisällöllä `"EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0 0\n2 10 0\nEOF"` palautetaan `np.array([[0,0],[10,0]])`. Testattu virheellisiä formaatteja (esim. ei-numeeriset koordinaatit, liian vähän osia koordinaattirivillä, ei-tuettu `EDGE_WEIGHT_TYPE`), jotka aiheuttavat virheen tai palauttavat tyhjän taulukon. Myös `FileNotFoundError` ja yleinen tiedostonlukuvirhe (`OSError`) on testattu.

**C. Lin-Kernighan-algoritmin ydinkomponentit (`test_lk_core.py`)**

Algoritmin sisäisiä ydinosia on testattu pääasiassa `simple_tsp_setup`-fixturea (5 solmun ongelma) hyödyntäen.
*   **`step()`-funktio:** Testattu, että funktio löytää yksinkertaisen 2-opt-parannuksen (`test_step_finds_simple_2_opt`). Varmistettu, ettei se tee muutoksia optimaaliseen kierrokseen (`test_step_no_improvement_on_optimal_tour`). Testattu myös reunatapauksia, kuten `BREADTH`-rajoitteen vaikutus (`test_step_hits_break_due_to_breadth_limit_zero`) ja tilanne, jossa tietylle `(t1, t2)`-parille ei löydy kandidaatteja (`test_step_no_candidates_for_t1_t2_pair`).
*   **`alternate_step()`-funktio:** Testattu parannusten löytäminen (`test_alternate_step_finds_improvement`) ja toiminta optimaalisella kierroksella (`test_alternate_step_no_improvement_on_optimal`). Testattu aikarajan ylittymisen käsittely (`test_alternate_step_deadline_exceeded`), `BREADTH_A/B`-rajoitteiden vaikutus (`test_alternate_step_restrictive_breadth`) ja kandidaattien ohittaminen, jos ne ovat `t1`, `t2` tai `chosen_y1` (`test_alternate_step_y2_candidate_continues`).
*   **`lk_search()`-funktio:** Testattu, että yksi `lk_search`-kierros pystyy parantamaan ei-optimaalisen kierroksen tunnettuun optimiin `simple_tsp_setup`-instanssissa (`test_lk_search_finds_optimum_for_simple_tsp`) eikä tee muutoksia optimaaliseen (`test_lk_search_no_improvement_on_optimal_tour`). Testattu aikarajojen käsittely: välitön paluu, jos aikaraja ylittynyt alussa (`test_lk_search_returns_none_if_deadline_at_start`), ja paluu, jos aikaraja ylittyy `step()`-kutsun jälkeen mutta ennen `alternate_step()`-kutsua (`test_lk_search_deadline_after_step_before_alternate`). Myös ei-parantavan `alternate_step`-tuloksen käsittely on testattu (`test_lk_search_handles_non_improving_alternate_step`).
*   **`lin_kernighan()`-pääalgoritmi:** Testattu algoritmin konvergoituminen tunnettuun optimiin `simple_tsp_setup`-instanssissa (`test_lin_kernighan_converges_on_simple_tsp`, `test_lin_kernighan_improves_to_optimum_simple_case`) ja toiminta, kun aloitetaan jo optimaalisella kierroksella (`test_lin_kernighan_no_change_on_optimal_tour`, `test_lin_kernighan_on_optimal_tour_simple_case`). Testattu toiminta pienillä instansseilla (2 ja 3 solmua). Aikarajojen noudattaminen on varmistettu (`test_lin_kernighan_respects_deadline`, `test_lin_kernighan_deadline_exceeded`, `test_lin_kernighan_respects_overall_deadline`). Testattu myös, että Delaunay-naapurit lasketaan sisäisesti, jos niitä ei anneta (`test_lin_kernighan_calculates_neighbors_if_not_provided`). `LK_CONFIG`-parametrien vaikutusta on myös testattu (`test_lin_kernighan_config_sensitivity`).

**D. Chained Lin-Kernighan -algoritmi (`test_chained_lk.py`)**

Iteratiivista `chained_lin_kernighan`-algoritmia on testattu.
*   **Konvergoituminen ja optimaalisuus:** Testattu konvergoituminen tunnettuun optimiin `simple_tsp_setup`-instanssissa (`test_chained_lin_kernighan_converges_on_simple_tsp`) ja aloitettaessa optimaalisella kierroksella (`test_chained_lin_kernighan_with_optimal_start`).
*   **Pysähtyminen `known_optimal_length` saavutettaessa:** Testattu useilla `rand`N-instansseilla (`rand4` - `rand12`), että algoritmi pysähtyy nopeasti löydettyään annetun optimipituuden, vaikka aikaraja olisi pidempi (`test_chained_lk_terminates_at_known_optimum`).
*   **Aikarajojen noudattaminen:** Testattu, että yleinen aikaraja pysäyttää algoritmin (`test_chained_lin_kernighan_respects_overall_deadline`).
*   **Iteraatioiden hallinta:** Testattu toiminta, kun `MAX_NO_IMPROVEMENT_ITERATIONS` saavutetaan (`test_chained_lin_kernighan_stops_on_max_no_improvement`). Myös yhden iteraation tapaus (`test_chained_lk_max_iterations_one`) ja useamman iteraation toiminta (`test_chained_lk_multiple_iterations`) on varmistettu.
*   **Reunatapaukset:** Testattu toiminta tyhjällä aloituskierroksella (`test_chained_lin_kernighan_empty_tour`).

**E. Työnkulun hallinta (`test_workflow.py`)**

Yksittäisen TSP-instanssin prosessointifunktiota `process_single_instance` on testattu.
*   **Virheiden käsittely:** Testattu tilanteet, joissa koordinaattien lataus epäonnistuu (`test_process_single_instance_no_coords_loaded`) tai optimaalista kierrosta ei löydy tiedostosta (`test_process_single_instance_no_opt_tour_loaded`). Myös `chained_lin_kernighan`-funktion sisäisen virheen (yleinen `Exception`) vaikutus tuloksiin on testattu (`test_process_single_instance_handles_chained_lk_exception`).

**F. Tulosten esitys (`test_output.py`)**

Tulosten visualisointi- ja yhteenvetofunktioita on testattu.
*   **`display_summary_table()`:** Testattu useilla reunatapauksilla, kuten tulosdatan ollessa tyhjä, sisältäessä vain virheellisiä tuloksia, tai kun `opt_len` tai `gap` puuttuvat tai ovat `inf` (`test_display_summary_table_edge_cases`).
*   **`plot_all_tours()`:** Testattu toiminta, kun tulosdata on tyhjä (`test_plot_all_tours_no_results`) tai ei sisällä piirrettäviä validia dataa (`test_plot_all_tours_no_valid_results_for_plotting`). Testattu myös varoituksen tulostuminen, jos piirrettävien kuvien määrä ylittää `MAX_SUBPLOTS_IN_PLOT` (`test_plot_all_tours_exceeds_max_subplots_prints_warning`, `test_plot_all_tours_num_to_plot_actual_is_zero`). Käyttämättömien subplotien akseleiden poistaminen (`set_axis_off`) on varmistettu (`test_plot_all_tours_turns_off_unused_subplots`). Onnistunut piirtopolku on testattu mockaamalla `matplotlib`-kutsut (`test_plot_all_tours_successful_path_mocked`).

## 3. Minkälaisilla syötteillä testaus tehtiin?

Testauksessa käytettiin monipuolisia syötteitä:
*   **Manuaalisesti määritellyt pienet instanssit:** Useita testejä varten luotiin pieniä, käsin laskettavissa olevia esimerkkejä solmujen koordinaateista ja kierroksista (esim. 2-5 solmua).
*   **`simple_tsp_setup`-fixture:** Tämä `conftest.py`-tiedostossa määritelty fixture tarjoaa konsistentin 5 solmun TSP-instanssin, johon kuuluu koordinaatit, etäisyysmatriisi, alkukierros, Delaunay-naapurit ja tunnettu optimaalinen kierros (`[0, 2, 4, 3, 1]`, kustannus n. `8.0006`). Tätä käytetään laajasti LK-ydinkomponenttien ja -algoritmien testauksessa.
*   **Reunatapaukset:** Tyhjät kierrokset, yhden tai kahden solmun kierrokset, virheelliset syötteet funktioille.
*   **Tiedostopohjaiset syötteet:** `read_opt_tour` ja `read_tsp_file` -funktioiden testauksessa luotiin dynaamisesti testitiedostoja (`tmp_path`-fixture) simuloimaan validia ja virheellistä TSPLIB-formaatin dataa.
*   **Ulkoiset TSPLIB-tiedostot:** `test_chained_lk_terminates_at_known_optimum` käyttää parametrisoidusti `rand4.tsp` ... `rand12.tsp` ja vastaavia `.opt.tour`-tiedostoja `verifications/random`-kansiosta.
*   **Algoritmin parametrit:** `LK_CONFIG`-sanakirjan eri asetuksia (esim. `MAX_LEVEL`, `BREADTH`, `TIME_LIMIT`, `MAX_NO_IMPROVEMENT_ITERATIONS`) muunneltiin testaamaan algoritmien käyttäytymistä eri rajoitteilla.

## 4. Miten testit voidaan toistaa?

Testit on kirjoitettu `pytest`-testauskehyksellä ja ne voidaan toistaa seuraavasti:
1.  **Asenna riippuvuudet:** Varmista, että Python-ympäristöön on asennettu `pytest`, `numpy` ja `scipy` (sekä `matplotlib` jos halutaan ajaa koodia, joka tuottaa visualisointeja, vaikka testit itsessään mockaavat sen).
2.  **Siirry projektin juurihakemistoon.**
3.  **Suorita `pytest`:** Aja komento `pytest` terminaalissa.
    *   Testit käyttävät pääosin dynaamisesti luotuja syötteitä tai `simple_tsp_setup`-fixturea, jotka eivät vaadi ulkoisia tiedostoja (lukuun ottamatta `randN`-testejä, jotka vaativat tiedostot `verifications/random`-kansiosta).
    *   Satunnaisuutta käytetään `double_bridge`-funktiossa, mutta testit on suunniteltu varmistamaan sen yleiset ominaisuudet. Yhdessä `double_bridge`-testissä käytetään `np.random.seed(42)` toistettavuuden varmistamiseksi.

## 5. Ohjelman toiminnan empiirisen testauksen tulokset graafisessa muodossa

Toimitetut yksikkötestit keskittyvät ohjelman eri osien oikeellisuuden varmistamiseen. Itse pääratkaisijaskripti (`lin_kernighan_tsp_solver.py`) sisältää toiminnallisuuden (`plot_all_tours` ja `display_summary_table`), joka visualisoi ja taulukoi tulokset useille TSPLIB-instansseille. Tämä toimii empiirisenä tulosten esityksenä algoritmin suorituskyvystä. Seuraavassa esitettävät kuvaajat on luotu 20 sekunnin aikarajalla ellei toisin ole mainittu.

### LK:n verfiointi brute force -algoritmin tuloksilla

[LK verifiointi random 20s](/images/lk_verifications_random_20s.png)

```Configuration parameters:
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
SUMMARY    21989.68 21989.68     0.00     0.00```

Skriptillä `/problems/create_tsp_problem.py` luotiin 4–12 solmun satunnaisia tsp-ongelmia, joiden optimaalinen reitti ratkaistiin skriptillä `/exact_tsp_solver/exact_tsp_solver.py` käymällä läpi kaikki mahdolliset reitit. LK algoritmin oikeellisuus pienillä solmujen määrällä varmistettiin ratkaisemalla pienet tsp-ongelmat ja vertaamalla tulosta optimaaliseen reittiin. Kuten kuvasta näkyy, niin LK-algoritmi löysi optimaalisen ratkaisun kaikkiin tsp-ongelmiin.  


## 6. Sovelluksen puutteet ja kehitysmahdollisuudet

Vaikka sovellus on testattu kattavasti ja saavuttaa hyvän testikattavuuden, on olemassa joitakin tunnistettuja puutteita ja mahdollisia kehityskohteita:

*   **Testikattavuus:** Vaikka 92% on hyvä, jäljellä olevat 8% koodista (lukuun ottamatta `if __name__ == '__main__':` -lohkoa) sisältävät pääasiassa monimutkaisempia ehtolauseita ja poikkeustilanteiden käsittelyä algoritmin ytimessä (`step`, `alternate_step`, `lk_search`). Näiden täydellinen kattaminen vaatisi erittäin spesifisiä ja mahdollisesti vaikeasti konstruoitavia testiskenaarioita.
    *   Esimerkiksi tietyt `continue`-lauseet syvällä sisäkkäisissä silmukoissa (`step`-funktiossa rivit 243, 338) tai harvinaisemmat poistumisehdot `alternate_step`-funktiossa (rivi 466, 482, 501) ovat vielä kattamatta.
    *   Myös `lin_kernighan`-funktion sisäinen `while True:` -silmukan poistumisehto rivillä 563 (kun `current_tour_obj.cost` ei parane) vaatisi tarkempaa testausta eri skenaarioissa.
    *   `lk_search`-funktion sisällä olevat `gain_threshold`-laskelmat ja niihin liittyvät ehdot (rivit 705-710) ovat osittain kattamatta.
*   **TSPLIB-formaatin tuki:** Tällä hetkellä `read_tsp_file` tukee vain `EUC_2D`-tyyppisiä etäisyysmatriiseja. Laajempi tuki muille TSPLIB-formaateille (esim. `GEO`, `ATT`, eksplisiittiset matriisit) parantaisi sovelluksen käytettävyyttä. Jokainen uusi tuettu formaatti vaatisi omat yksikkötestinsä.
*   **Naapurilistojen generointi:** Vain Delaunay-triangulaatioon perustuva naapurilista on implementoitu. Muiden naapuristrategioiden (esim. k-lähimmät naapurit, quadrant-naapurit) lisääminen voisi parantaa ratkaisun laatua tietyissä instansseissa. Nämä vaatisivat omat testinsä.
*   **Parametrien viritys:** `LK_CONFIG`-sanakirjan parametrit ovat globaaleja ja niiden optimaaliset arvot voivat vaihdella instanssityypin ja -koon mukaan. Kehittyneempi parametrien hallinta tai automaattinen viritys voisi olla hyödyllistä. Tämän testaaminen olisi monimutkaista ja vaatisi todennäköisesti laajempaa empiiristä testausta.
*   **Suorituskyky suurilla instansseilla:** Vaikka algoritmi on tehokas, Pythonin luontainen hitaus verrattuna käännettyihin kieliin voi rajoittaa suorituskykyä erittäin suurilla TSP-instansseilla (useita tuhansia solmuja). Kriittisten osien (esim. `Tour`-luokan operaatiot, etäisyyslaskenta) optimointi tai siirtäminen Cythoniin voisi olla kehityssuunta. Suorituskykytestaus ja profilointi auttaisivat tunnistamaan pullonkauloja.
*   **Käyttöliittymä ja käytettävyys:** Nykyinen sovellus on komentorivipohjainen. Graafinen käyttöliittymä tai parempi integrointi muihin työkaluihin voisi parantaa käytettävyyttä.
*   **Virheiden raportointi:** Vaikka virheitä käsitellään, käyttäjälle annettava palaute virhetilanteissa (esim. tiedostojen lukuvirheet) voisi olla yksityiskohtaisempaa ja ohjaavampaa.
*   **Dokumentaatio:** Koodin sisäistä dokumentaatiota (docstringejä) voisi paikoin laajentaa selkeyden parantamiseksi, erityisesti monimutkaisimmissa algoritmin osissa. Myös käyttäjädokumentaatiota voisi laajentaa.

Nämä kohdat tarjoavat suuntaviivoja sovelluksen jatkokehitykselle ja laadun parantamiselle.