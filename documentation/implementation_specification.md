# Toteutusdokumentti

Tämä dokumentti kuvaa Lin-Kernighan-heuristiikkaan perustuvan TSP-ratkaisijan (Traveling Salesperson Problem) Python-toteutuksen.

## 1. Ohjelman yleisrakenne

Ohjelma on toteutettu Pythonilla ja hyödyntää ulkoisia kirjastoja, kuten `numpy` numeriseen laskentaan, `matplotlib` tulosten visualisointiin ja `scipy` (erityisesti `Delaunay`-triangulaatioon) naapurilistojen generointiin.

Ohjelman pääkomponentit ovat:

1.  **Konfiguraatio ja vakiot:**
    *   `TSP_FOLDER_PATH`: Polku TSPLIB-instanssitiedostoihin.
    *   `FLOAT_COMPARISON_TOLERANCE`: Toleranssi liukulukujen vertailuun.
    *   `MAX_SUBPLOTS_IN_PLOT`: Maksimimäärä alikaavioita tulosten visualisoinnissa.
    *   `LK_CONFIG`: Sanakirja, joka sisältää Lin-Kernighan-algoritmin parametrit. Näihin kuuluvat `MAX_LEVEL` (rekursion maksimisyvyys `step`-funktiossa), `BREADTH` (haun leveys `step`-funktion eri tasoilla), `BREADTH_A`, `BREADTH_B`, `BREADTH_D` (haun leveydet `alternate_step`-funktion eri vaiheissa) sekä oletusaikarajat.

2.  **`Tour`-luokka:**
    *   Edustaa kierrosta (solmujen permutaatiota).
    *   Sisältää metodit kierroksen tehokkaaseen käsittelyyn:
        *   `__init__`: Alustaa kierroksen solmujärjestyksestä ja laskee tarvittaessa sen pituuden etäisyysmatriisin `D` avulla.
        *   `init_cost`: Laskee ja tallentaa kierroksen kokonaiskustannuksen.
        *   `next`, `prev`: Palauttavat annetun solmun seuraajan ja edeltäjän kierroksella (O(1) -operaatiot `pos`-taulukon ansiosta).
        *   `sequence`: Tarkistaa, onko solmu `b` polulla solmusta `a` solmuun `c`.
        *   `flip`: Kääntää osan kierroksesta (segmentin).
        *   `get_tour`: Palauttaa kierroksen listana, normalisoituna alkamaan solmusta 0, jos mahdollista.
        *   `flip_and_update_cost`: Kääntää segmentin ja päivittää kierroksen kustannuksen tehokkaasti (2-opt-muutoksen kustannusdelta).

3.  **Aputoiminnot:**
    *   `build_distance_matrix`: Laskee painotetun etäisyysmatriisin koordinaateista (Euklidinen etäisyys).
    *   `delaunay_neighbors`: Generoi Delaunay-triangulaatioon perustuvat naapurilistat, joita käytetään ehdokassiirtojen rajaamiseen LK-algoritmissa.
    *   `double_bridge`: Toteuttaa "double bridge"-kick-siirron (4-opt-siirto) kierroksen satunnaiseksi rikastamiseksi ja paikallisen minimin välttämiseksi.

4.  **Lin-Kernighan-algoritmin ydinlogiikka:**
    *   `step`: Rekursiivinen funktio, joka toteuttaa Lin-Kernighan-algoritmin ydinaskeleen (Applegate et al., Algoritmi 15.1). Tutkii k-opt-siirtoja (standardi- ja Mak-Morton-tyyppisiä) kierroksen parantamiseksi.
    *   `alternate_step`: Toteuttaa vaihtoehtoisen ensimmäisen askeleen LK-algoritmissa (Applegate et al., Algoritmi 15.2), joka tutkii tiettyjä 3-opt- ja 5-opt-siirtoja.
    *   `lk_search`: Yksittäinen Lin-Kernighan-hakukierros (Applegate et al., Algoritmi 15.3). Yrittää löytää parantavan siirtosarjan käyttämällä `step`- ja `alternate_step`-funktioita.
    *   `lin_kernighan`: Pääfunktio Lin-Kernighan-heuristiikalle (Applegate et al., Algoritmi 15.4). Soveltaa iteratiivisesti `lk_search`-funktiota merkityistä solmuista, kunnes parannusta ei enää löydy tai aikaraja saavutetaan.

5.  **Kokonaisratkaisu ja tiedostonkäsittely:**
    *   `chained_lin_kernighan`: Toteuttaa ketjutetun Lin-Kernighan-metaheuristiikan (Applegate et al., Algoritmi 15.5). Toistaa LK-ajoja `double_bridge`-kick-siirtoja välttääkseen paikallisia minimejä. Pysähtyy aikarajaan tai jos tunnettu optimi löydetään.
    *   `read_opt_tour`: Lukee optimaalisen kierroksen `.opt.tour`-tiedostosta (TSPLIB-muoto).
    *   `read_tsp_file`: Lukee TSP-ongelman koordinaatit `.tsp`-tiedostosta (tukee vain `EUC_2D`-tyyppiä).
    *   `process_single_instance`: Käsittelee yhden TSP-ongelman: lataa datan, suorittaa `chained_lin_kernighan`-algoritmin ja laskee tilastot (kuten etäisyyden optimaaliseen ratkaisuun).
    *   `display_summary_table`: Tulostaa yhteenvedon käsiteltyjen instanssien tuloksista.
    *   `plot_all_tours`: Visualisoi heuristiset ja (jos saatavilla) optimaaliset kierrokset.

6.  **Pääohjelma (`if __name__ == '__main__':`)**
    *   Iteroi `TSP_FOLDER_PATH`-kansiossa olevien `.tsp`-tiedostojen yli.
    *   Kullekin instanssille kutsuu `process_single_instance`.
    *   Lopuksi kutsuu `display_summary_table` ja `plot_all_tours` näyttääkseen kootut tulokset.

## 2. Saavutetut aika- ja tilavaativuudet

Lin-Kernighan-heuristiikan tarkkaa teoreettista aika- ja tilavaativuutta on vaikea määrittää sen heuristisen luonteen vuoksi. Vaativuus riippuu instanssista ja algoritmin parametreista.

**Aikavaativuus (arviot):**

*   **`build_distance_matrix`**: $O(n^2)$, missä n on solmujen määrä, koska kaikki solmuparien väliset etäisyydet lasketaan.
*   **`delaunay_neighbors`**: SciPy:n Delaunay-triangulaatio on tyypillisesti $O(n * log(n))$ 2D-tapauksessa.
*   **`Tour`-luokan operaatiot:**
    *   `next`, `prev`: $O(1)$ `pos`-taulukon ansiosta.
    *   `flip`: $O(k)$, missä k on käännettävän segmentin pituus (pahimmillaan $O(n)$ ).
    *   `flip_and_update_cost`: $O(k)$ segmentin kääntämiselle, $O(1)$ kustannuksen päivitykselle (2-opt).
*   **`step` ja `alternate_step`:** Näiden funktioiden kompleksisuus on merkittävä. Ne iteroivat naapureiden yli (Delaunay rajoittaa määrää). Rekursiosyvyys (`MAX_LEVEL`) ja haun leveys (`BREADTH`) vaikuttavat suorituskykyyn. Karkeasti arvioiden yhden `step`-kutsun kompleksisuus voisi olla luokkaa $O(MAXLEVEL * BREADTH * n * C)$, missä C liittyy naapurien käsittelyyn ja etäisyyslaskuihin.
*   **`lk_search`**: Kutsuu `step`- ja `alternate_step`-funktioita. Sen kompleksisuus on verrannollinen näiden funktioiden kompleksisuuteen.
*   **`lin_kernighan`**: Iteroi, kunnes parannusta ei löydy. Yhden iteraation aikana kutsutaan `lk_search` enintään `n` kertaa (merkityille solmuille). Iterointien määrä on instanssiriippuvainen.
*   **`chained_lin_kernighan`**: Suorittaa `lin_kernighan`-funktion useita kertoja, joiden välissä on $O(n)$-kompleksisuuden `double_bridge`-kick. Kokonaiskestoa rajoittaa annettu aikaraja.
*   **Kokonaisaikavaativuus:** Empiirisesti Lin-Kernighan-heuristiikan on raportoitu skaalautuvan usein luokkaa $O(n^{2.2})$ - $O(n^3)$ tyypillisillä euklidisilla instansseilla, mutta tämä ei ole tiukka teoreettinen yläraja. Toteutuksen suorituskykyä dominoivat `step`- ja `alternate_step`-funktioiden sisäiset silmukat ja rekursio.

**Tilavaativuus:**

*   **`coords`**: $O(n)$ koordinaateille.
*   **`D` (etäisyysmatriisi)**: $O(n^2)$. Tämä on usein dominoiva tekijä.
*   **`neigh` (naapurilistat)**: $O(n * kavg)$, missä kavg on keskimääräinen Delaunay-naapureiden määrä. Pahimmillaan $O(n^2)$, mutta käytännössä paljon vähemmän (lähellä $O(n)$ tasomaisille graafeille).
*   **`Tour`-olio**: `order`- ja `pos`-taulukot vaativat $O(n)$ tilaa.
*   **Rekursiopino (`step`)**: Syvyys enintään `MAX_LEVEL`.
*   **Kokonais_tilavaativuus**: Pääasiassa $O(n^2)$ etäisyysmatriisin vuoksi.

## 3. Suorituskyky- ja O-analyysivertailu

*   **Teoreettinen vs. Käytännön suorituskyky:** Vaikka tarkkaa O-notaatiota on vaikea antaa koko algoritmille, sen komponenttien analyysi (esim. $O(n^2)$ etäisyysmatriisille) auttaa ymmärtämään pullonkauloja. LK:n vahvuus on sen erinomainen empiirinen suorituskyky monissa TSP-instansseissa, vaikka teoreettiset takuut puuttuvat.
*   **Tietorakenteiden vaikutus:** `Tour`-luokan `pos`-taulukko mahdollistaa $O(1)$-aikaiset `next`- ja `prev`-kyselyt, mikä on kriittistä algoritmin tehokkuudelle.
*   **Delaunay-naapurien käyttö:** Rajoittamalla ehdokassiirrot Delaunay-naapureihin vähennetään merkittävästi tutkittavien siirtojen määrää verrattuna kaikkien mahdollisten naapureiden tarkasteluun, mikä parantaa suorituskykyä erityisesti suurilla instansseilla.
*   **Parametrien (`MAX_LEVEL`, `BREADTH`) vaikutus:** Nämä parametrit tarjoavat kompromissin ratkaisun laadun ja laskenta-ajan välillä. Suuremmat arvot voivat johtaa parempiin ratkaisuihin, mutta lisäävät laskenta-aikaa.

## 4. Työn mahdolliset puutteet ja parannusehdotukset

**Puutteet:**

1.  **Rajoitettu tuki etäisyystyypeille:** Ohjelma tukee tällä hetkellä vain `EUC_2D`-tyyppisiä TSPLIB-tiedostoja.
2.  **Kiinteät ja globaalit parametrit:** `LK_CONFIG`-asetukset ovat globaalisti määriteltyjä ja kiinteitä ajon aikana. Dynaamisesti mukautuvat parametrit voisivat parantaa suorituskykyä, ja parametrien välittäminen funktioille globaalin muuttujan sijaan voisi parantaa modulaarisuutta ja testattavuutta monimutkaisemmissa käyttötapauksissa.
3.  **Yksinkertainen siirto:** `double_bridge` on yleinen kick-menetelmä, mutta kehittyneempiä tai vaihtelevia strategioita voitaisiin harkita.
4.  **Ei rinnakkaistusta:** Algoritmin suoritus voitaisiin mahdollisesti nopeuttaa rinnakkaistamalla esimerkiksi `lk_search`-kutsuja eri aloituspisteille tai ajamalla useita `chained_lin_kernighan`-ketjuja rinnakkain.
5.  **Virheidenkäsittely tiedostojen lukemisessa:** Vaikka virheitä käsitellään, käyttäjälle annettava palaute voisi olla yksityiskohtaisempaa (esim. `read_opt_tour` ja `read_tsp_file` voisivat antaa tarkempia virheilmoituksia kommentoitujen `print`-kutsujen sijaan).
6.  **Kustannusten päivitys `step`-funktiossa:** `step`-funktio kumuloi `delta`-arvoa (saavutettua hyötyä) sen sijaan, että laskisi kierroksen kokonaiskustannuksen jokaisen väliaikaisen käännön jälkeen uudelleen. Tämä on tehokasta, mutta vaatii huolellisuutta logiikassa. Kuitenkin, kun parantava sekvenssi löydetään, se sovelletaan alkuperäiseen kierrokseen `flip_and_update_cost`-metodilla, joka laskee kustannusmuutokset tarkasti 2-opt-siirroille.

**Parannusehdotukset:**

1.  **Laajempi TSPLIB-tuki:** Lisätään tuki muille etäisyysfunktioille (esim. `GEO`, `ATT`).
2.  **Adaptiiviset parametrit:** Kehitetään mekanismeja, jotka säätävät `MAX_LEVEL`- ja `BREADTH`-parametreja dynaamisesti ongelman ominaisuuksien tai haun edistymisen perusteella.
3.  **Kehittyneemmät ehdokasstrategiat:** Otetaan käyttöön kehittyneempiä ehdokaslistojen generointimenetelmiä (esim. alpha-läheisyys).
4.  **Monipuolisemmat siirrot:** Kokeillaan ja implementoidaan muita kick-mekanismeja `chained_lin_kernighan`-algoritmiin.
5.  **Rinnakkaistus:** Tutkitaan mahdollisuuksia rinnakkaistaa laskentaa (esim. käyttämällä `multiprocessing`-kirjastoa).
6.  **Koodin profilointi ja optimointi:** Profiloidaan koodi säännöllisesti ja optimoidaan kriittisimpiä osia.
7.  **Käyttöliittymä:** Graafinen käyttöliittymä voisi helpottaa ohjelman käyttöä ja tulosten analysointia.
