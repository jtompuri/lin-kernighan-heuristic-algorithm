# Viikkoraportti 3

## 1. Mitä olen tehnyt tällä viikolla?

Tämän viikon tehtäviä:
- Testit LK-algoritmin ytimelle, `Tour`-luokalle, apufunktioille ja kokonaisille työkuluille (workfow)
- Testien esiin nostamien ongelmien ratkominen
- Algoritmin verifiointi pienillä tsp-ongelmilla ja TSPLIB95-kirjaston ongelmilla, joihin tunnetaan optimaalinen ratkaisu
- Testausdokumentin ja toteutusdokumentin laatiminen 

## 2. Miten ohjelma on edistynyt?

Pääsin edellisellä viikolla yksinkertaista tsp-ratkaisijaa parempiin tuloksiin eli noin 5 %:n päähän optimaalisesta ratkaisusta. Testien perusteella löysin kaksi bugia `flip`-metodista, jotka ratkaisemalla LK-algoritmi sai ratkaistua kaikki pienet tsp-ongelmat ja pääsi 0-2 %:n päähän optimaalisesta ratkaisusta TSPLIB95-ongelmissa. Tämä oli selvä läpimurto algoritmin toiminnassa. Tämän jälkeen olen keskittynyt parantamaan LK-algoritmin vakautta testien avulla. Pidin huolta, että ratkaisujen laatu ei laske, kun tein testien perusteella muutoksia koodiin.

## 3. Mitä opin tällä viikolla / tänään?

Suurin osa työajasta on mennyt testitapausten laatimiseen, joten olen oppinut paljon yksikkötestauksesta ja testikattavuudesta. 

## 4. Mikä jäi epäselväksi tai tuottanut vaikeuksia? 

TSPLIB95-tuloksissa oli kaksi poikkeustapausta, joissa gap jäi korkeaksi (50 % ja 30 %). Sama toistui pidemmällä 900 sekunnin aikarajalla, joten en ollut varma, jääkö algoritmi jumiin paikalliseen minimiin vai onko kyseisten ongelmien ratkaiseminen vain hidasta. Tarkastelin lähemmin toista tapausta pitkällä aikarajalla (7200 s.) ja pääsin lähelle optimaalista ratkaisua myös tässä tapauksessa, joten päättelin LK-algoritmin toimivan oikein ja välttävän paikallisia minimejä.

## 5. Mitä teen seuraavaksi?

Seuraavia työvaiheita:
- LK-algoritmin parametrien testaaminen (jos jää aikaa) 
- Toteutusdokumentin viimeistely
- Testausdokumentin viimeistely

## Kolmannen viikon viikon tuloksia

LK-algoritmin tulokset ennen `flip`-metodin bugin korjausta:

![Ennen korjausta](/images/lin_kernighan_tsp_solver_plots_20s.png)

Tulokset bugien korjauksen jälkeen:

![Korjauksen jälkeen](/images/lin_kernighan_tsp_solver_verifications_tsplib95_20s.png)
