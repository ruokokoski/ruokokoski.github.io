# Transformerit: modernin tekoälyn perusta

Transformer-arkkitehtuuri [1] mullisti luonnollisen kielen käsittelyn antamalla malleille kyvyn ymmärtää asiayhteyksiä. Tämä mahdollisti suurten kielimallien kehityksen ja johti ChatGPT:n kaltaisten sovellusten syntyyn. Kielimallien menestyksen myötä transformereita on alettu soveltamaan monilla muillakin aloilla. Niiden käyttö on laajentunut esimerkiksi konenäköön, äänenkäsittelyyn, videoanalyysiin, kuvien generointiin, biologisten sekvenssien analyysiin ja aikasarjojen ennustamiseen [2]. Nämä sovelluskohteet osoittavat Transformer-mallin joustavuuden ja soveltuvuuden moniin eri tarkoituksiin.

## Transformer-mallin edut

Ennen transformereita syötteitä käsiteltiin pääasiassa toistuvilla neuroverkoilla (engl. *recurrent neural network*, RNN) ja pitkillä lyhytkestomuisteilla (engl. *long short-term memory*, LSTM). Ne käsittelevät syötteitä askel kerrallaan, mikä on hidasta. Sarjamuotoinen tietojenkäsittely ei myöskään hyödynnä tehokkaasti moderneja grafiikkaprosessoreita (engl. *graphic processing unit*, GPU), joita käytetään neuroverkkojen laskentatehtävissä. Transformer-malli puolestaan käsittelee koko syötteen kerralla rinnakkain ja hyödyntää siten paremmin grafiikkaprosessorien tarjoamaa laskentatehoa [2]. Rinnakkaislaskenta tehostaa oppimista erityisesti suurten datamäärien käsittelyssä. Se parantaa myös mallien skaalautuvuutta ja mahdollistaa erittäin suurten mallien käytön.

Transformerien vahvuus on sen huomiomekanismi (engl. *attention*, [3]), joka mahdollistaa kaikkien syötteen osien samanaikaisen tarkastelun. Huomiomekanismin avulla malli arvioi eri tietoelementtien (esimerkiksi sanojen tai kuvien osien) merkitystä suhteessa toisiinsa ja etsii samankaltaisuuksia syötteen eri osien välillä. Eri osia painotetaan niiden tärkeyden mukaan, minkä avulla tunnistetaan, mitkä osat syötteestä ovat olennaisimpia ymmärtämisen kannalta. Transformerit eivät tee oletuksia syötteen rakenteesta, vaan pystyy joustavasti käsittelemään pitkiäkin riippuvuuksia. Transformer-malli onkin syrjäyttänyt perinteiset neuroverkkomallit monissa sovelluksissa.

Mallin yksi merkittävä etu on myös sen tehokkuus siirto-oppimisessa (engl. *transfer learning*). Malli voidaan aluksi kouluttaa suurilla datamäärillä itseohjautuvasti, jonka jälkeen malli voidaan hienosäätää valituille erityistehtäville ohjatun oppimisen avulla [2]. Siirtovaikutus on erityisen hyödyllistä tilanteissa, joissa käytettävissä oleva datamäärä on rajallinen. Tällöin malli voidaan opettaa ensin jollain muulla vain osittain vastaavalla tiedolla. Siirtovaikutuksen ansiosta malli voi silti saavuttaa korkean suorituskyvyn hyödyntämällä aiemmin esikoulutuksessa oppimiaan tietoja.

Transformerit ovat osoittautuneet monipuoliseksi eri sovellusalueilla ilman suuria rakenteellisia muutoksia. Huomiomekanismi mahdollistaa erilaisten sekvenssityyppien käsittelyn, koska syötteet voivat olla epäsäännöllisiä vektorijoukkoja, järjestettyjä sekvenssejä tai muita yleisempiä esitysmuotoja. Tämä tekee mallista soveltuvan laajaan valikoimaan tehtäviä.

## Perusrakenne ja toiminta

Alkuperäinen Transformer-malli [1] kehitettiin konekäännöksiä varten. Malli koostuu kahdesta pääkomponentista: kooderista (engl. *encoder*), joka käsittelee syötteen ja dekooderista (engl. *decoder*), joka tuottaa käännöksen. 

Kuvassa 1 esitetään alkuperäisen Transformer-arkkitehtuurin rakenne. Kielen kääntämisessä merkkijonot pilkotaan ensin pienempiin yksiköihin, kuten sanoihin, alisanoihin tai merkkeihin (tokenisointi). Nämä tokenit muutetaan vektorimuotoisiksi esityksiksi, joita kutsutaan sanaupotuksiksi (engl. *word embeddings*). Sanaupotuksiin lisätään paikkakoodaus (engl. *positional encoding*), minkä jälkeen ne syötetään kooderiin, joka tuottaa tokeneille kontekstuaaliset esitykset. Toisin sanoen nämä esitykset sisältävät tietoa sanojen merkityksistä ja suhteista toisiinsa huomioiden samalla kontekstin, jossa sanat esiintyvät. Dekooderi puolestaan käyttää näitä esityksiä sekä aiemmin tuotettuja sanoja tuottaakseen ulostulon yksi sana kerrallaan. Malli laskee kullekin sanalle todennäköisyydet koko sanaston osalta ja valitsee seuraavan sanan todennäköisimpien vaihtoehtojen joukosta.

![Kuva 1. Alkuperäinen Transformer-arkkitehtuuri, joka sisältää kooderin ja dekooderin.](/pics/transformer_malli.png)

*Kuva 1. Alkuperäinen Transformer-arkkitehtuuri, joka sisältää kooderin ja dekooderin (muokattu lähteestä Vaswani, 2017 [1]).*

Eri sovelluksissa voidaan hyödyntää erilaisia arkkitehtuurin kokoonpanoja, riippuen siitä millaisesta tehtävästä on kyse. Alkuperäinen kooderi-dekooderi-malli soveltuu etenkin sekvenssistä sekvenssiin (engl. *sequence-to-sequence*) tehtäviin, kuten kielen kääntämiseen [1]. Pelkkä kooderi on puolestaan tehokas sekvenssin ymmärtämistehtävissä, kuten luokittelussa ja tunnistuksessa. Vastaavasti pelkkä dekooderi soveltuu sekvenssien luontiin, kuten kielen mallintamiseen. Esimerkiksi hyvin suosittuja GPT-malleja (engl. *Generative Pre-trained Transformer*) käytetään tekstin luomisessa ja ne toimivat pelkästään dekoodereina [4].

### Kooderi ja dekooderi
Transformer-arkkitehtuurissa kooderi ja dekooderi koostuvat useista identtisistä kerroksista, joita voidaan pinota. Alkuperäisessä mallissa sekä kooderilla että dekooderilla oli kuusi kerrosta, mutta käytännössä kerroksia voidaan lisätä tarpeen mukaan. Jokaisessa kerroksessa hyödynnetään monipäistä itsehuomiota (engl. *multi-head self-attention*) ja eteenpäin syöttävää neuroverkkoa (engl. *feedforward neural network*, FFNN).

Kooderi vastaa syötteen käsittelystä ja sen muuntamisesta. Vaikka kooderikerrokset ovat muuten identtisiä, jokaisella kerroksella on omat opitut parametrit. Kunkin kooderikerroksen rakenne sisältää itsehuomiomekanismin, eteenpäin syöttävän neuroverkon, jäännösyhteyden (engl. *residual connection*) ja kerroksen normalisoinnin (engl. *layer normalization*).

Jäännösyhteys luo oikotien eri kerrosten välille. Yhteyden avulla syötteen alkuperäiset ominaisuudet voidaan yhdistää kunkin kerroksen tuottamaan uuteen tietoon. Tämä vähentää oppimisprosessiin liittyvää gradienttien katoamista ja helpottaa syvien verkkojen koulutusta [5]. Kerrosnormalisointi puolestaan parantaa oppimisprosessin tehokkuutta ja lisää mallin vakautta tasapainottamalla kerroksien aktivaatioita [6].

Ennen kuin syöte siirtyy kooderin käsittelyyn, se käy läpi esikäsittelyvaiheet, jotka sisältävät sanaupotukset ja paikkakoodauksen. Sanaupotuksessa syötteen osat muutetaan matemaattiseen muotoon vektoreiksi, joissa samankaltaiset syötteen osat sijaitsevat lähellä toisiaan vektoreiden moniulotteisessa esityksessä. Esimerkiksi sanat "metsä" ja "puu" saattavat olla toisiaan muistuttavia vektoreita, kun taas täysin erilaisten sanojen, kuten "koira" ja "kvanttifysiikka", vektorietäisyys on suurempi. Sanaupotusten perusidea pätee myös muunlaiseen dataan, kuten kuviin ja ääniin. Myös niissä syötteen osat muutetaan vektoreiksi, joiden piirteet mahdollistavat eri osien vertailun. Paikkakoodaus puolestaan lisää syötteeseen tietoa osien sijainnista, jolloin malli pystyy tunnistamaan ja hyödyntämään osien suhteellisia sijainteja.

Kooderi luo syötteestä vektoreita, jotka esittävät merkityksellisiä riippuvuuksia syötteen eri osien välillä. Kooderikerroksissa itsehuomiomekanismi tunnistaa ja painottaa tärkeitä suhteita syötteen sisällä, jolloin malli pystyy ymmärtämään sekä lähekkäin että kaukana sijaitsevien osien välisiä yhteyksiä.

Dekooderin päätehtävä transformereissa on tuottaa sarjamuotoista dataa, mikä tekee siitä sopivan sekvenssien generointitehtäviin, kuten kielenkäännökseen ja tekstin tuottamiseen. Se hyödyntää kooderilta saatuja tietoja ja aiemmin generoitua sisältöä luodakseen seuraavan osan peräkkäisestä jaksosta vaihe vaiheelta.  

Dekooderin itsehuomiokerros eroaa hieman kooderin vastaavasta. Dekooderi ei saa nähdä tulevia osia generoitavasta jaksosta, mikä estetään käyttämällä peitettyä itsehuomiomekanismia (engl. *masked self-attention*). Tämä tarkoittaa, että dekooderi voi tarkastella vain aiemmin luotuja osia nykyisen osan tuottamiseen. Jos dekooderi saisi myös ennustettavat osat syötteenä, se voisi huijata oppimisprosessin aikana hyödyntämällä tulevia tietoja. Tällöin mallin kyky oppia ennustamaan seuraavia osia heikkenisi merkittävästi. 

Dekooderi hyödyntää peitetyn itsehuomiomekanismin lisäksi erillistä huomiomekanismia, joka yhdistää dekooderin tuottaman tiedon kooderin muokkaamaan syötteeseen. Näin dekooderi pystyy tuottamaan johdonmukaista sisältöä. Dekooderin viimeiset kerrokset koostuvat lineaarisesta kerroksesta ja sitä seuraavasta softmax-funktiosta, joka laskee ennustettavien vaihtoehtojen todennäköisyydet. Dekooderi liittää tuottamansa sisällön jatkuvasti kasvavaan syötteeseensä, kunnes malli ennustaa loppumerkin, jolloin prosessi päättyy.

### Huomiomekanismi
Huomiomekanismi on transformereiden toiminnan perusta. Alun perin huomiomekanismi esiteltiin RNN-mallien laajennuksena parantamaan niiden suorituskykyä kielenkäännöksessä [3], mutta sen todellinen läpimurto tapahtui vasta Transformer-malleissa. Toisin kuin perinteiset neuroverkot, joissa syötteitä kerrotaan kiinteillä painoilla, transformer käyttää syötteestä riippuvia painokertoimia, jotka mukautuvat dynaamisesti syötteen mukaan [2].

Huomiomekanismissa kaikki syötevektorit järjestetään riveittäin syötematriisiin **X**, jonka koko on *n* × *d*, missä  
- *n* on syötteen vektorien määrä (esimerkiksi sanojen lukumäärä tekstissä)  
- *d* on kunkin vektorin dimensio (piirteiden määrä).

Matriisi X muunnetaan kolmeksi erilliseksi matriisiksi:  
- kyselyt **Q** (koko *n* × *dq*)  
- avaimet **K** (koko *n* × *dk*)  
- arvot **V** (koko *n* × *dv*)  

Tämä tapahtuu kertomalla syötematriisi **X** kolmella eri painomatriisilla **Wq**, **Wk** ja **Wv**, jotka ovat mallin koulutuksen aikana opittuja parametreja. On tärkeää, että matriisien koot ovat yhteensopivia kertolaskua varten.

Tämän muunnoksen jälkeen jokainen syötteen osa saa oman kysely-, avain- ja arvovektorin.  
- Kyselyt (**Q**) määrittävät, mitä tietoa malli kulloinkin etsii.  
- Avaimet (**K**) koodaavat syötteen osien keskeiset ominaisuudet, joita kyselyt käyttävät vertailukohtana.  
- Arvot (**V**) sisältävät tiedon, joka lopulta yhdistyy mallin lopulliseen tulokseen.  

Tämä rakenne mahdollistaa sen, että malli voi joustavasti painottaa eri syöteosien merkitystä.

Syöteosien välisten riippuvuuksien laskeminen tapahtuu kysely- ja avainvektorien pistetulon avulla, joka mittaa vektorien samankaltaisuutta. Pistetulon suuruus ilmaisee, kuinka vahva yhteys kahden syöteosan välillä on. 
Jokaisen syöteosan kyselyvektori kerrotaan kaikkien muiden syöteosien avainvektoreiden kanssa, jolloin saadaan huomioarvot, jotka kuvaavat kunkin syöteosan merkitystä suhteessa muihin syöteosiin. 

Jotta pistetulojen arvot pysyisivät hallinnassa erityisesti silloin, kun vektoreiden ulottuvuus *d* on suuri, ne skaalataan jakamalla vektoreiden ulottuvuuden neliöjuurella. Ilman skaalausta pistetulojen arvot voivat kasvaa niin suuriksi, että mallin herkkyys kärsii ja mallista tulee vaikea kouluttaa [8].
Skaalauksen jälkeen pistetuloille suoritetaan normalisointi softmax-funktion avulla, joka suhteuttaa ne välille 0 – 1.
Nämä huomioarvot ilmaisevat, kuinka paljon painoarvoa kullekin syöteosalle annetaan.

Lopuksi huomioarvot kerrotaan vielä arvovektoreilla, jolloin jokaisen syöteosan merkitys painotetaan sen saamien huomioarvojen perusteella. Lopputuloksena saadaan uusi matriisi, joka huomioi syöteosien väliset riippuvuudet. Tämä mahdollistaa sen, että malli pystyy keskittymään olennaisiin osiin riippumatta siitä, ovatko ne lähekkäin vai kaukana toisistaan.

Skaalatun pistetulohuomion laskeminen voidaan esittää yhtälöllä [1]:

<img src="/pics/attention.png" alt="Attention" width="70%">

Koska syötevektorit kootaan matriisimuotoon, huomion laskeminen voidaan suorittaa rinnakkain. Laskennassa voidaan hyödyntää grafiikkaprosessoreita ja kaikki syöteosat voidaan käsitellä samanaikaisesti. Tämä parantaa merkittävästi laskennan nopeutta ja mahdollistaa suurempien datamäärien käsittelyn.

Kuva 3 havainnollistaa skaalatun pistetulohuomion laskentaa. Tämä rakenne muodostaa yksittäisen huomiopään monipäisessä huomiomekanismissa.

<img src="/pics/huomio.png" alt="Kuva 3. Skaalatun pistetulohuomion laskeminen" width="65%">

*Kuva 3. Skaalatun pistetulohuomion laskeminen (muokattu lähteestä Bishop, 2024 [2]).*

### Monipäinen huomio
Transformerit käyttävät usein monipäistä huomiomekanismia, vaikka periaatteessa yksi huomiokerros voisi riittää. Jos käytössä on vain yksi huomiokerros, se voi tunnistaa tietyn riippuvuuden syötteen osien välillä, mutta saattaa sivuuttaa muita tärkeitä suhteita. 

Monipäisessä huomiossa huomiomekanismeja lisätään rinnakkain, mikä mahdollistaa useiden eri näkökulmien samanaikaisen tutkimisen. Monipäisen huomiomekanismin rakenne on esitetty alla olevassa kuvassa 4. Koska eri huomiopäät (engl. *attention heads*) käyttävät omia painomatriisejaan **Q**-, **K**- ja **V**-matriisien laskemiseen, jokainen pää pystyy käsittelemään erityyppisiä riippuvuuksia syötteen osien välillä. Lopullinen huomioarvo muodostetaan yhdistämällä kaikkien päiden antamat tulokset. Näin malli pystyy ymmärtämään syvällisemmin syötettä ja kykenee monimutkaisempiin tehtäviin.

![Kuva 4. Monipäinen huomio koostuu useasta rinnakkaisesta huomiokerroksesta](/pics/monipainen.png)

*Kuva 4. Monipäinen huomio koostuu useasta rinnakkaisesta huomiokerroksesta (muokattu lähteestä Vaswani, 2017 [1]).*

### Eteenpäin syöttävä neuroverkko
Monipäisen huomiomekanismin jälkeen Transformer-arkkitehtuurissa on tavallinen eteenpäin syöttävä neuroverkko (FFNN). Verkko käsittelee jokaisen syötevektorin lisäten syötteeseen epälineaarisia muunnoksia. Huomiomekanismin tuottamaa informaatiota siis jatkojalostetaan ennen sen siirtämistä seuraavaan kerrokseen. Tämä prosessi parantaa mallin kykyä oppia monimutkaisempia riippuvuuksia syötteiden välillä [2].

Eteenpäin syöttävä neuroverkko koostuu tyypillisesti kahdesta täysin liitetystä kerroksesta sekä epälineaarisesta aktivointifunktiosta, kuten ReLU (engl. *rectified linear unit*, [7]). Vaikka FFNN on rakenteeltaan varsin yksinkertainen, se on olennainen osa Transformer-mallin suorituskyvyn kannalta.

### Paikkakoodaus
Syötteen osien järjestys on oleellinen tieto useimmissa peräkkäisiä käsittelytehtäviä vaativissa sovelluksissa. Esimerkiksi kielimalleissa lauseen merkitys voi muuttua täysin toiseksi rippuen siitä, missä järjestyksessä sanat esiintyvät. Transformer-mallissa ei kuitenkaan itsessään ole sisäänrakennettua ymmärrystä järjestyksestä. Järjestyksen ja suhteellisten etäisyyksien mallintaminen on ratkaistava lisäämällä erillinen paikkakoodaus syötteeseen [1]. Ilman tätä mekanismia malli ei voisi tunnistaa järjestystä tai suhteita syötteen eri osien välillä, mikä tarkoittaisi, että se tuottaisi saman tuloksen, vaikka osien järjestys vaihtuisi.

Alkuperäisessä Transformer-artikkelissa [1] käytettiin *sin*- ja *cos*-funktioita paikkakoodauksena. Jokaiselle syötteen elementille laskettiin paikkakoodaus seuraavasti: 

- Parillisille indekseille: `PE(pos, 2i) = sin(pos / 10000^(2i / d_model))`  
- Parittomille indekseille: `PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))`  

missä `pos` on elementin sijainti ja `d_model` mallin dimensio. Tällainen paikkakoodaus mahdollistaa syötteen elementtien suhteiden mallintamisen ja toimii eri pituisilla syötteillä.

Nykyisin suurimmassa osassa kielimalleja käytetään opittuja paikkakoodauksia (engl. *learned positional embeddings*), mutta pitkän kontekstin malleissa, kuten LLaMA, yleistyvät myös kiertävät paikkakoodaukset (engl. *rotary positional embeddings (RoPE)*).

## Yhteenveto

Transformerit muuttivat tekoälyn kehityksen suunnan kertaheitolla. Huomiomekanismin ansiosta ne pystyvät tarkastelemaan koko syötettä rinnakkain ja löytämään olennaiset suhteet dynaamisesti. Tämä oivallus avasi tien suurille kielimalleille ja loi perustan järjestelmille, jotka kykenevät tuottamaan, tulkitsemaan ja muokkaamaan dataa ennennäkemättömällä tasolla.

Arkkitehtuurin voima ei rajoitu pelkästään tekstiin. Sama rakenne toimii yhtä hyvin kuvissa, äänissä, biologisissa sekvensseissä ja aikasarjoissa, toisin sanoen missä tahansa, missä data muodostaa rakenteita ja riippuvuuksia. Transformereiden skaalautuvuus ja rinnakkaislaskennan hyödyntäminen ovat tehneet niistä koko modernin tekoälyn moottorin.

Transformerit ovat koko nykyisen tekoälyn perusta, tekninen läpimurto, joka ohjaa todennäköisesti myös seuraavan sukupolven innovaatioita.

## Lähdeviitteet

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. ja Polosukhin, I. (2017). ”Attention is all you need”. *Advances in Neural Information Processing Systems*, s. 5998–6008.

[2] Bishop, C. M. ja Bishop, H. (2024). *Deep learning: Foundations and concepts.* Springer Nature

[3] Bahdanau, D., Cho, K. ja Bengio, Y. (2015). ”Neural machine translation by jointly learning to align and translate”. Teoksessa: *3rd International Conference on Learning Representations, ICLR 2015.*

[4] Radford, A. (2018). *Improving language understanding by generative pre-training.* Technical Report. OpenAI.

[5] He, K., Zhang, X., Ren, S. ja Sun, J. (2016). ”Deep residual learning for image recognition”. Teoksessa: *Proceedings of the IEEE conference on computer vision and pattern recognition*, s. 770–778.

[6] Ba, J. L. (2016). ”Layer normalization”. *arXiv preprint arXiv:1607.06450.*

[7] Glorot, X., Bordes, A. ja Bengio, Y. (2011). ”Deep sparse rectifier neural networks”. Teoksessa: *Proceedings of the fourteenth international conference on artificial intelligence and statistics.* JMLR Workshop ja Conference Proceedings, s. 315–323.

