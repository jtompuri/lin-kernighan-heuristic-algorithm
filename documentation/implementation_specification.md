# Toteutusdokumentti

Tämä dokumentti kuvaa Lin-Kernighan-heuristiikkaan perustuvan TSP-ratkaisijan (Traveling Salesperson Problem) lopullisen Python-toteutuksen.

## 1. Ohjelman yleisrakenne

Ohjelma on toteutettu Python-pakettina (`lin_kernighan_tsp_solver`) ja hyödyntää ulkoisia kirjastoja, kuten `numpy` numeriseen laskentaan, `matplotlib` tulosten visualisointiin ja `scipy` (erityisesti `Delaunay`-triangulaatioon ja etäisyysfunktioihin) naapurilistojen generointiin.

### Pakettien rakenne:

*   **`config.py`:** Konfiguraatiot ja vakiot
*   **`lk_algorithm.py`:** Lin-Kernighan-algoritmin ydinlogiikka  
*   **`starting_cycles.py`:** Aloituskierrosten generointialgoritmit
*   **`tsp_io.py`:** TSPLIB-tiedostojen lukeminen
*   **`utils.py`:** Apufunktiot (tulostus, visualisointi)
*   **`main.py`:** Pääohjelma ja rinnakkaistus
*   **`__main__.py`:** Komentorivirajapinta

### Keskeiset konfiguraatiot (`config.py`):

1.  **Konfiguraatio ja vakiot:**
    *   `TSP_FOLDER_PATH`: Polku TSPLIB-instanssitiedostoihin.
    *   `SOLUTIONS_FOLDER_PATH`: Polku heurististen kierrosten tallentamiseen.
    *   `FLOAT_COMPARISON_TOLERANCE`: Toleranssi liukulukujen vertailuun.
    *   `MAX_SUBPLOTS_IN_PLOT`: Maksimimäärä alikaavioita tulosten visualisoinnissa.
    *   `LK_CONFIG`: Sanakirja, joka sisältää Lin-Kernighan-algoritmin parametrit. Näihin kuuluvat `MAX_LEVEL` (rekursion maksimisyvyys `step`-funktiossa), `BREADTH` (haun leveys `step`-funktion eri tasoilla), `BREADTH_A`, `BREADTH_B`, `BREADTH_D` (haun leveydet `alternate_step`-funktion eri vaiheissa), `TIME_LIMIT` (aikaraja), `STARTING_CYCLE` (aloituskierrosalgoritmi) sekä `SAVE_TOURS` (tallennusasetus).
    *   `STARTING_CYCLE_CONFIG`: Aloituskierrosalgoritmien konfiguraatio, joka sisältää saatavilla olevat metodit (`natural`, `random`, `nearest_neighbor`, `greedy`, `boruvka`, `qboruvka`) ja niiden parametrit.

2.  **`Tour`-luokka (`lk_algorithm.py`):**
    *   Edustaa kierrosta (solmujen permutaatiota) tehokkaasti.
    *   Sisältää metodit kierroksen tehokkaaseen käsittelyyn:
        *   `__init__`: Alustaa kierroksen solmujärjestyksestä ja laskee tarvittaessa sen pituuden etäisyysmatriisin `D` avulla. Ylläpitää `pos`-taulukkoa nopeiden `next`/`prev`-operaatioiden mahdollistamiseksi.
        *   `init_cost`: Laskee ja tallentaa kierroksen kokonaiskustannuksen.
        *   `next`, `prev`: Palauttavat annetun solmun seuraajan ja edeltäjän kierroksella (O(1) -operaatiot `pos`-taulukon ansiosta).
        *   `sequence`: Tarkistaa, onko solmu `b` polulla solmusta `a` solmuun `c`.
        *   `flip`: Kääntää osan kierroksesta (segmentin) tehokkaasti **paikallaan (in-place)** optimoiduilla modulo-operaatioilla, mikä vähentää muistinvarausta ja parantaa suorituskykyä.
        *   `get_tour`: Palauttaa kierroksen listana, normalisoituna alkamaan solmusta 0, jos mahdollista.
        *   `flip_and_update_cost`: Kääntää segmentin ja päivittää kierroksen kustannuksen tehokkaasti (2-opt-muutoksen kustannusdelta).

3.  **Aloituskierrosten generointialgoritmit (`starting_cycles.py`):**
    *   `generate_starting_cycle`: Pääfunktio, joka valitsee sopivan aloituskierrosalgoritmin.
    *   `_natural_tour`: Luonnollinen järjestys (0, 1, 2, ..., n-1) - nopein vaihtoehto.
    *   `_random_tour`: Satunnainen permutaatio.
    *   `_nearest_neighbor_tour`: Lähimmän naapurin heuristiikka.
    *   `_greedy_tour`: Ahne kaarten valinta (lyhimmät kaaret ensin).
    *   `_boruvka_tour`: Borůvkan MST-pohjainen rakentaminen.
    *   `_qboruvka_tour`: Nopea Borůvka (Concorden oletusmetodi) 2-opt-parannuksilla.

4.  **Aputoiminnot (`lk_algorithm.py`):**
    *   `build_distance_matrix`: Laskee painotetun etäisyysmatriisin koordinaateista (Euklidinen etäisyys).
    *   `delaunay_neighbors`: Generoi Delaunay-triangulaatioon perustuvat naapurilistat, joita käytetään ehdokassiirtojen rajaamiseen LK-algoritmissa.
    *   `double_bridge`: Toteuttaa "double bridge"-kick-siirron (4-opt-siirto) kierroksen satunnaiseksi rikastamiseksi ja paikallisen minimin välttämiseksi.
    *   `SearchContext`: Dataclass, joka pitää hakukontekstin (etäisyysmatriisi, naapurit, kustannukset, aikaraja).

5.  **Lin-Kernighan-algoritmin ydinlogiikka (`lk_algorithm.py`):**
    *   `step`: Rekursiivinen funktio, joka toteuttaa Lin-Kernighan-algoritmin ydinaskeleen (Applegate et al., Algoritmi 15.1). Tutkii k-opt-siirtoja (standardi- ja Mak-Morton-tyyppisiä) kierroksen parantamiseksi. Sisältää optimointeja kuten aikarajojen määräaikaistarkistukset ja metodiviittausten välimuistittaminen.
    *   `alternate_step`: Toteuttaa vaihtoehtoisen ensimmäisen askeleen LK-algoritmissa (Applegate et al., Algoritmi 15.2), joka tutkii tiettyjä 3-opt- ja 5-opt-siirtoja.
    *   `lk_search`: Yksittäinen Lin-Kernighan-hakukierros (Applegate et al., Algoritmi 15.3). Yrittää löytää parantavan siirtosarjan käyttämällä `step`- ja `alternate_step`-funktioita. **Varmistaa, että `alternate_step`-funktion löytämä siirtosarja on aidosti parantava ennen sen palauttamista.**
    *   `lin_kernighan`: Pääfunktio Lin-Kernighan-heuristiikalle (Applegate et al., Algoritmi 15.4). Soveltaa iteratiivisesti `lk_search`-funktiota merkityistä solmuista, kunnes parannusta ei enää löydy tai aikaraja saavutetaan.

6.  **Kokonaisratkaisu ja metaheuristiikka (`lk_algorithm.py`):**
    *   `chained_lin_kernighan`: Toteuttaa ketjutetun Lin-Kernighan-metaheuristiikan (Applegate et al., Algoritmi 15.5). Toistaa LK-ajoja `double_bridge`-kick-siirtoja välttääkseen paikallisia minimejä. Pysähtyy aikarajaan tai jos tunnettu optimi löydetään. **Suorittaa lopuksi kierroksen kustannuksen uudelleenlaskennan varmistaakseen tuloksen oikeellisuuden.**
    *   `_perform_kick_and_lk_run`: Suorittaa kick-siirron ja LK-ajon yhdistelmän.
    *   `_check_for_optimality`: Tarkistaa, onko tunnettu optimaalinen ratkaisu löydetty.
    *   `_apply_and_update_best_tour`: Soveltaa parantavan sekvenssin ja päivittää parhaan kierroksen.

7.  **Tiedostonkäsittely ja I/O (`tsp_io.py`):**
    *   `read_opt_tour`: Lukee optimaalisen kierroksen `.opt.tour`-tiedostosta (TSPLIB-muoto).
    *   `read_tsp_file`: Lukee TSP-ongelman koordinaatit `.tsp`-tiedostosta (tukee vain `EUC_2D`-tyyppiä).

8.  **Tulostus ja visualisointi (`utils.py`):**
    *   `display_summary_table`: Tulostaa yhteenvedon käsiteltyjen instanssien tuloksista.
    *   `plot_all_tours`: Visualisoi heuristiset ja (jos saatavilla) optimaaliset kierrokset.
    *   `save_heuristic_tour`: Tallentaa heuristisen kierroksen tiedostoon.

9.  **Pääohjelma ja rinnakkaistus (`main.py`):**
    *   `main`: Pääfunktio, joka tukee sekä rinnakkaista että peräkkäistä käsittelyä.
    *   `process_single_instance`: Käsittelee yhden TSP-ongelman: lataa datan, generoi aloituskierroksen valitulla algoritmilla, suorittaa `chained_lin_kernighan`-algoritmin ja laskee tilastot (kuten etäisyyden optimaaliseen ratkaisuun).
    *   Rinnakkaistus: Käyttää `concurrent.futures.ProcessPoolExecutor`-luokkaa moniydin-käsittelyyn.

10. **Komentorivirajapinta (`__main__.py`):**
    *   Argparse-pohjainen komentorivirajapinta.
    *   Tukee eri aloituskierrosalgoritmeja, aikarajoja, rinnakkaistusasetuksia ja kierrosten tallennusta.

## 2. Saavutetut aika- ja tilavaativuudet

Lin-Kernighan-heuristiikan tarkkaa teoreettista aika- ja tilavaativuutta on vaikea määrittää sen heuristisen luonteen vuoksi. Vaativuus riippuu instanssista ja algoritmin parametreista.

**Aikavaativuus (arviot):**

*   **`build_distance_matrix`**: $O(n^2)$, missä n on solmujen määrä, koska kaikki solmuparien väliset etäisyydet lasketaan.
*   **`delaunay_neighbors`**: SciPy:n Delaunay-triangulaatio on tyypillisesti $O(n \cdot \log \cdot n)$ 2D-tapauksessa.
*   **Aloituskierrosten algoritmeista:**
    *   `_natural_tour`: $O(n)$ - nopein vaihtoehto.
    *   `_random_tour`: $O(n)$ - satunnaispermutaatio.
    *   `_nearest_neighbor_tour`: $O(n^2)$ - klassinen heuristiikka.
    *   `_greedy_tour`: $O(n^2 \log n)$ - kaarten järjestäminen ja syklintarkistus.
    *   `_boruvka_tour`: $O(n^2 \log n)$ - MST:n rakentaminen ja DFS-kierto.
    *   `_qboruvka_tour`: $O(n^2 \log n + k \cdot n^2)$, missä k on 2-opt-parannusiterointien määrä.
*   **`Tour`-luokan operaatiot:**
    *   `next`, `prev`: $O(1)$ `pos`-taulukon ansiosta.
    *   `flip`: $O(k)$, missä k on käännettävän segmentin pituus (pahimmillaan $O(n)$ ).
    *   `flip_and_update_cost`: $O(k)$ segmentin kääntämiselle, $O(1)$ kustannuksen päivitykselle (2-opt).
*   **`step` ja `alternate_step`:** Näiden funktioiden kompleksisuus on merkittävä. Ne iteroivat naapureiden yli (Delaunay rajoittaa määrää). Rekursiosyvyys (`MAX_LEVEL`) ja haun leveys (`BREADTH`) vaikuttavat suorituskykyyn. Karkeasti arvioiden yhden `step`-kutsun kompleksisuus voisi olla luokkaa $O(MAX\_LEVEL \times BREADTH \times n \times C)$, missä C liittyy naapurien käsittelyyn ja etäisyyslaskuihin.
*   **`lk_search`**: Kutsuu `step`- ja `alternate_step`-funktioita. Sen kompleksisuus on verrannollinen näiden funktioiden kompleksisuuteen.
*   **`lin_kernighan`**: Iteroi, kunnes parannusta ei löydy. Yhden iteraation aikana kutsutaan `lk_search` enintään `n` kertaa (merkityille solmuille). Iterointien määrä on instanssiriippuvainen.
*   **`chained_lin_kernighan`**: Suorittaa `lin_kernighan`-funktion useita kertoja, joiden välissä on $O(n)$-kompleksisuuden `double_bridge`-kick. Kokonaiskestoa rajoittaa annettu aikaraja.
*   **Rinnakkaistus**: Pääohjelma tukee rinnakkaistusta usean CPU-ytimen hyödyntämiseksi, mikä voi merkittävästi parantaa kokonaissuorituskykyä useita instansseja käsiteltäessä.
*   **Kokonaisaikavaativuus:** Empiirisesti Lin-Kernighan-heuristiikan on raportoitu skaalautuvan usein luokkaa $O(n^{2.2})$ - $O(n^3)$ tyypillisillä euklidisilla instansseilla, mutta tämä ei ole tiukka teoreettinen yläraja. Toteutuksen suorituskykyä dominoivat `step`- ja `alternate_step`-funktioiden sisäiset silmukat ja rekursio.

**Tilavaativuus:**

*   **`coords`**: $O(n)$ koordinaateille.
*   **`D` (etäisyysmatriisi)**: $O(n^2)$. Tämä on usein dominoiva tekijä muistinkulutuksessa.
*   **`neigh` (naapurilistat)**: $O(n \cdot k_{avg})$, missä $k_{avg}$ on keskimääräinen Delaunay-naapureiden määrä. Pahimmillaan $O(n^2)$, mutta käytännössä paljon vähemmän (lähellä $O(n)$ tasomaisille graafeille).
*   **`Tour`-olio**: `order`- ja `pos`-taulukot vaativat $O(n)$ tilaa.
*   **Aloituskierrosten algoritmit**: Useimmat vaativat $O(n)$ - $O(n^2)$ lisätilaa väliaikaisille tietorakenteille.
*   **Rekursiopino (`step`)**: Syvyys enintään `MAX_LEVEL`.
*   **Rinnakkaistus**: Jokainen työprosessi tarvitsee oman kopion tiedoista, mikä lisää muistinkulutusta.
*   **Kokonais_tilavaativuus**: Pääasiassa $O(n^2)$ etäisyysmatriisin vuoksi.

## 3. Suorituskyky- ja O-analyysivertailu

*   **Teoreettinen vs. Käytännön suorituskyky:** Vaikka tarkkaa O-notaatiota on vaikea antaa koko algoritmille, sen komponenttien analyysi (esim. $O(n^2)$ etäisyysmatriisille) auttaa ymmärtämään pullonkauloja. LK:n vahvuus on sen erinomainen empiirinen suorituskyky monissa TSP-instansseissa, vaikka teoreettiset takuut puuttuvat.
*   **Tietorakenteiden vaikutus:** `Tour`-luokan `pos`-taulukko mahdollistaa $O(1)$-aikaiset `next`- ja `prev`-kyselyt, mikä on kriittistä algoritmin tehokkuudelle. `flip`-metodin toteutus paikallaan (in-place) optimoiduilla modulo-operaatioilla vähentää muistinvarausta ja parantaa suorituskykyä verrattuna väliaikaisia listoja luovaan toteutukseen.
*   **Aloituskierrosalgoritmien vaikutus:** Valittu aloituskierrosalgoritmi vaikuttaa merkittävästi sekä alkuperäiseen ratkaisun laatuun että LK-algoritmin konvergoitumisnopeuteen. `qboruvka` on oletusvalinta Concorde-solverin mukaisesti, tarjoten hyvän kompromissin nopeuden ja laadun välillä.
*   **Delaunay-naapurien käyttö:** Rajoittamalla ehdokassiirrot Delaunay-naapureihin vähennetään merkittävästi tutkittavien siirtojen määrää verrattuna kaikkien mahdollisten naapureiden tarkasteluun, mikä parantaa suorituskykyä erityisesti suurilla instansseilla.
*   **Parametrien (`MAX_LEVEL`, `BREADTH`) vaikutus:** Nämä parametrit tarjoavat kompromissin ratkaisun laadun ja laskenta-ajan välillä. Suuremmat arvot voivat johtaa parempiin ratkaisuihin, mutta lisäävät laskenta-aikaa.
*   **Rinnakkaistuksen hyödyt:** Useamman TSP-instanssin käsittely rinnakkain voi merkittävästi parantaa kokonaissuorituskykyä, erityisesti nykyaikaisilla moniytimisillä järjestelmillä.
*   **Optimoinnit:** Toteutuksessa on useita suorituskykyoptimointeja, kuten metodiviittausten välimuistitusta, määräaikaisia aikarajojen tarkistuksia `step`-funktiossa ja tehokkaita modulo-operaatioita `flip`-metodissa.

## 4. Työn mahdolliset puutteet ja parannusehdotukset

**Puutteet:**

1.  **Rajoitettu tuki etäisyystyypeille:** Ohjelma tukee tällä hetkellä vain `EUC_2D`-tyyppisiä TSPLIB-tiedostoja.
2.  **Globaalit konfiguraatiot:** `LK_CONFIG` ja muut asetukset ovat globaalisti määriteltyjä, mikä voi hankaloittaa useamman konfiguraation samanaikaista käyttöä.
3.  **Kiinteä kick-strategia:** `double_bridge` on ainoa käytössä oleva kick-menetelmä ketjutetussa algoritmissa.
4.  **Yksinkertainen optimaalisuuden tarkistus:** Optimaalisuuden tarkistus perustuu vain yksinkertaiseen toleranssiksi vertailuun.
5.  **Rajoitettu konfiguroitavuus ajon aikana:** Vaikka komentoriviparametrit mahdollistavat jotain konfigurointia, monet algoritmin sisäiset parametrit ovat kiinteitä.

**Parannusehdotukset:**

1.  **Laajempi TSPLIB-tuki:** Lisätään tuki muille etäisyysfunktioille (esim. `GEO`, `ATT`, `MANHATTAN`).
2.  **Adaptiiviset parametrit:** Kehitetään mekanismeja, jotka säätävät `MAX_LEVEL`- ja `BREADTH`-parametreja dynaamisesti ongelman ominaisuuksien tai haun edistymisen perusteella.
3.  **Kehittyneemmät ehdokasstrategiat:** Otetaan käyttöön kehittyneempiä ehdokaslistojen generointimenetelmiä (esim. alpha-läheisyys, quadrant-based candidates).
4.  **Monipuolisemmat kick-strategiat:** Kokeillaan ja implementoidaan muita kick-mekanismeja `chained_lin_kernighan`-algoritmiin (esim. Or-opt, 3-opt kicks).
5.  **Hybridimenetelmät:** Yhdistetään LK muihin heuristiikkoihin (esim. genetic algorithms, simulated annealing).
6.  **Koodin profilointi ja optimointi:** Profiloidaan koodi säännöllisesti ja optimoidaan kriittisimpiä osia. Harkitaan Cython- tai Numba-optimointeja pullonkaulakohtiin.
7.  **Käyttöliittymä:** Graafinen käyttöliittymä voisi helpottaa ohjelman käyttöä ja tulosten analysointia.
8.  **Muistinhallinnan optimointi:** Suurille instansseille voitaisiin implementoida muistintehokkaampia tietorakenteita tai etäisyysmatriisin osittaislaskentaa.
9.  **Parempi virheidenkäsittely:** Robustimpi virheidenkäsittely ja palautuminen epätavallisista tilanteista.
10. **Testikattavuuden parantaminen:** Vaikka testikattavuus on korkea, jotkin edge-caset ja harvinaiset algoritmin haarat voisivat hyötyä lisätesteistä.

## 5. Laajojen kielimallien käyttö

Harjoitustyön toteutuksessa on käytetty Googlen Gemini Pro 2.5 (Preview) laajaa kielimallia (LLM). Kielimallia on käytetty ratkaisujen ideointiin, koodikatselmointeihin, refaktorointiin sekä testien luomiseen.

Kokemukset kielimallin käytöstä ohjelmoinnissa ovat olleet suurelta osin positiivisia. Aiemmin olen käyttänyt ohjelmointiongelmien malliratkaisujen etsimiseen esimerkiksi verkosta löytyviä lähteitä kuten Stackoverflow'ta. Triviaalien ongelmien kohdalla kielimalli korvasi lähes kokonaan verkkolähteet. Kielimallin käyttö nopeuttaa ohjelmointia, mutta edellyttää ihan yhtä lailla koodin säännöllistä refaktorointia toisteisuuden poistamiseksi, hyvien ohjelmointikäytäntöjen noudattamiseksi ja käyttämättömän koodin siivoamiseksi pois. 

**Lin-Kernighan-algoritmin pääkehitys:**
Lin-Kernighan pääalgoritmin kehittämisessä kielimallista oli enemmän haittaa kuin hyötyä. Kielimalli tarjoaa LK-algoritmista hyvin yksinkertaisen toteutuksen, joka ei vastaa lainkaan algoritmin TSP-kirjan mallitoteutusta. Kielimallin avulla voi tuottaa Simple TSP Solverin kaltaisen 2-opt-vaihtoja tekevän algoritmin, mutta sen kutsuminen LK-algoritmiksi on harhaanjohtavaa. Käytin LK-algoritmin toteutukseen tukena TSP-kirjan pseudokoodia, jota pyrin tulkitsemaan parhaani mukaan.

**Aloituskierrosalgoritmit:**
Aloituskierrosalgoritmien toteutuksessa kielimalli oli erittäin hyödyllinen. Se auttoi ymmärtämään eri algoritmien (lähimmän naapurin, ahne, Borůvka, QBorůvka) toimintaperiaatteita ja tarjosi tehokkaita toteutustapoja. Kielimalli osasi myös ehdottaa sopivia optimointeja suurille instansseille.

**Testien kehittäminen:**
Testitapausten ideoinnissa ja toteutuksessa kielimalli vähensi paljon toisteista ohjelmointia. Koin myös, että en olisi luultavasti keksinyt kaikkia poikkeustapauksia testeihin. Kielimalli ei osaa kehittää testejä varten TSP-ongelmia, jotka varmasti johtavat tiettyyn algoritmin haaraan. Tämä on myös ihmiselle hyvin haasteellista ja tältä osin tukeuduin tunnettujen ja tutkittujen TSP-ongelmien piirteisiin. Tämä ei kuitenkaan aina tuottanut tulosta, minkä vuoksi 100 %:n testikattavuus on vaikea saavuttaa järkevällä työmäärällä.

**Koodin rakenne ja optimoinnit:**
Kielimalli oli hyödyllinen projektin rakenteen suunnittelussa, modulaaristen komponenttien eriyttämisessä ja suorituskykyoptimointien tunnistamisessa. Se auttoi myös rinnakkaistuksen implementoinnissa ja komentorivirajapinnan kehittämisessä.

**Kokonaisarvio:**
Kielimallin käyttö oli kokonaisuudessaan hyödyllistä, erityisesti infrastruktuurin ja tukifunktioiden kehittämisessä. Kriittisen algoritmin ytimen kehittämisessä perinteinen lähestymistapa (kirjallisuus, pseudokoodi) osoittautui luotettavammaksi.

## 6. Käyttöohje

### Asennus ja riippuvuudet

Projekti vaatii Python 3.9+ ja seuraavat kirjastot:
- numpy
- scipy
- matplotlib

Riippuvuudet voi asentaa komennolla:
```bash
pip install -r requirements.txt
```

### Käyttö

#### Peruskäyttö
```bash
python -m lin_kernighan_tsp_solver
```

#### Parametrien säätäminen
```bash
# Käytä tiettyä aloituskierrosalgoritmia
python -m lin_kernighan_tsp_solver --starting-cycle qboruvka

# Aseta aikaraja
python -m lin_kernighan_tsp_solver --time-limit 20.0

# Käytä peräkkäistä käsittelyä
python -m lin_kernighan_tsp_solver --sequential

# Älä tallenna kierroksia
python -m lin_kernighan_tsp_solver --no-save-tours

# Käsittele vain tietyt tiedostot
python -m lin_kernighan_tsp_solver problems/berlin52.tsp problems/eil51.tsp
```

#### Saatavilla olevat aloituskierrosalgoritmit
- `natural`: Luonnollinen järjestys (nopein)
- `random`: Satunnaispermutaatio
- `nearest_neighbor`: Lähimmän naapurin heuristiikka
- `greedy`: Ahne kaarten valinta
- `boruvka`: Borůvkan MST-pohjainen
- `qboruvka`: Nopea Borůvka (oletus, Concorden mukaisesti)

### Tulokset

Ohjelma tulostaa:
1. **Yhteenvetotaulukko**: Instanssien tulokset (optimaalinen vs. heuristinen pituus, gap, aika)
2. **Visualisointi**: Kierrosten kuvaajat (optimaaliset ja heuristiset reitit)
3. **Tallennetut kierrokset**: `solutions/`-kansioon (jos `SAVE_TOURS=True`)

## 7. Lähteet

Applegate, David L. & Bixby, Robert E. & Chvtal,  Vaek & Cook, William J. (2006): *The Traveling Salesman Problem : A Computational Study*, Princeton University Press.

Lin, Shen & Kernighan, Brian W. (1973): ”An Effective Heuristic Algorithm for the Traveling-Salesman Problem”, Operations Research, Vol. 21, No. 2, s. 498–516.

Reinelt, Gerhard (1995): *TSPLIB95: TSP Library*, University of Heidelberg, Operations Research Group. Saatavilla osoitteessa: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/