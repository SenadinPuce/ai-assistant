"""Golden Q&A set for evaluating answer quality of the CRAG assistant.

Kept as Python data (not JSON) so each example can carry inline sourcing notes.
Extend this list as the ingested knowledge base grows — questions should be
answerable from documents actually uploaded to the vector store.
"""

QA_EXAMPLES: list[dict] = [
    {
        "question": "Sta je teorija sistema i kako je nastala?",
        "reference": (
            "Teorija sistema je nauka koja se bavi izucavanjem sistema i zakonitostima "
            "koje u njima vladaju. Nastala je iz potrebe pronalazenja naucnih i prakticnih "
            "metoda za analizu i rjesavanje problema koje tradicionalne metode ne rjesavaju "
            "zadovoljavajuce."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Dovrsi definiciju iz poglavlja 1.2: 'Sistem je skup vise elemenata...' i navedi koje ulazne, izlazne i okolinske velicine se spominju.",
        "reference": (
            "Sistem je skup vise elemenata koji su medjusobno povezani i imaju interaktivno "
            "dejstvo. Ima ulazne i izlazne fizicke velicine, kao i uticaje okoline, a u sistemu "
            "se vrsi transformacija ulaza u izlaze i moze postojati povratna veza."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Sta je stanje sistema i zasto je vazno za predvidjanje buduceg ponasanja?",
        "reference": (
            "Stanje sistema je skup podataka o ponasanju sistema u datom presjeku vremena, "
            "sa ciljem predvidjanja njegovog buduceg ponasanja. Sto imamo vise zabiljezenih "
            "stanja, to je jasnija slika o ponasanju sistema u vremenu."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Sta je proces u kontekstu sistema?",
        "reference": (
            "Proces je dejstvo sistema, odnosno aktivan rad i postupak promjene stanja elemenata "
            "sistema u vremenu, pri cemu promjena treba dati potreban i dovoljan efekat."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Kako se sistemi dijele prema prirodi nastanka?",
        "reference": (
            "Prema prirodi nastanka sistemi se dijele na prirodne, tehnicke i organizacione "
            "sisteme. Prirodni su van direktnog uticaja covjeka, tehnicki su prirodni sistemi "
            "pod kontrolom covjeka, a organizacioni kombinuju tehnicki sistem i covjeka."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Kako se sistemi dijele prema promjeni stanja u vremenu?",
        "reference": (
            "Prema promjeni stanja u vremenu sistemi se dijele na staticke i dinamicke. "
            "Staticki se ne mijenjaju u vremenu, dok se kod dinamickih stanje mijenja u funkciji "
            "vremena uz transformaciju i prenos energije, materije i/ili informacija."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Sta je poslovni sistem i koju ulogu ima prema proizvodnim sistemima?",
        "reference": (
            "Poslovni sistem je skup proizvodnih, ekonomskih i drustvenih podsistema i elemenata "
            "koji povezuju okolinu (trziste) sa proizvodnim sistemima. To je slozen i jedinstven "
            "dinamicki sistem koji objedinjuje mehanizme trzista, rada, proizvodnje, finansiranja, "
            "upravljanja i kontrole."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Kako je definisan proizvodni sistem i koji su njegovi glavni podsistemi?",
        "reference": (
            "Proizvodni sistem je skup tehnickih, informacionih i energetskih struktura i "
            "ucesnika procesa rada organizovanih za ostvarivanje funkcije cilja i planiranih "
            "efekata. Cesto se opisuje kroz podsisteme kao sto su priprema rada, transport, "
            "skladistenje, kontrola, odrzavanje, upravljanje i snabdijevanje."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Sta je tehnoloski sistem, a sta obradni sistem?",
        "reference": (
            "Tehnoloski sistem je skup obradnih sistema koji omogucava izvodjenje svih operacija "
            "obrade za odredjeni proizvod, odnosno transformaciju sirovine u gotov proizvod. "
            "Obradni sistem izvodi konkretnu operaciju obrade i obuhvata upravljanje, rad i kontrolu."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Sta predstavlja tehnoloski proces?",
        "reference": (
            "Tehnoloski proces je dio proizvodnog procesa i podrazumijeva postupnu promjenu "
            "oblika, dimenzija, spoljasnjeg izgleda ili unutrasnje strukture materijala ili "
            "polufabrikata radi dobijanja gotovog proizvoda odgovarajuceg kvaliteta."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Kako glasi hijerarhija poslovnog, proizvodnog, tehnoloskog i obradnog sistema?",
        "reference": (
            "Hijerarhija je: poslovni sistem je najvisi nivo, zatim proizvodni, potom tehnoloski, "
            "a obradni sistem je najuzi nivo. Visi nivo obuhvata vise nizih podsistema."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "U poglavlju 1.8, koje su osnovne vrste tehnoloskih procesa prema strukturi toka (jednostavni, analiticki, sinteticki, mjesoviti)?",
        "reference": (
            "Prema strukturi toka tehnoloski procesi se dijele na jednostavne, analiticke, "
            "sinteticke i mjesovite procese. Mjesoviti se u praksi javljaju kao kombinacije "
            "analiticko-sintetickih i sinteticko-analitickih tokova."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Sta karakterise jednostavni tehnoloski proces?",
        "reference": (
            "Jednostavni tehnoloski proces je pretezno jednolinijski i predstavlja preradu jedne "
            "sirovine radi dobijanja materijala, poluproizvoda ili proizvoda."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Prema poglavlju 1.8, objasni analiticki tehnoloski proces kao razlaganje jedne sirovine na vise izlaza i navedi tipicnu primjenu (nafta/gas, petrohemija).",
        "reference": (
            "Analiticki tehnoloski proces podrazumijeva da se od jedne sirovine dobija vise "
            "materijala, poluproizvoda i gotovih proizvoda. Tipican je za preradu nafte i gasa "
            "u petrohemijskoj industriji."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Sta je sinteticki tehnoloski proces i koji je tipican primjer?",
        "reference": (
            "Sinteticki tehnoloski proces znaci sklapanje dijelova u podsklopove i sklopove "
            "do gotovog proizvoda. Tipican je za masinogradnju, npr. montazu automobila, "
            "aviona ili brodova."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Objasni razliku izmedju poslovnog i proizvodnog sistema.",
        "reference": (
            "Poslovni sistem je siri pojam koji objedinjuje proizvodne, ekonomske i drustvene "
            "podsisteme te veze sa trzistem i okolinom. Proizvodni sistem je uzi i odnosi se na "
            "tehnicko-organizacioni sistem koji realizuje proizvodni proces i moze biti sastavni "
            "dio poslovnog sistema."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Objasni razliku izmedju tehnoloskog i obradnog sistema.",
        "reference": (
            "Tehnoloski sistem obuhvata vise obradnih sistema i omogucava izvodjenje svih "
            "operacija obrade potrebnih za proizvod. Obradni sistem je pojedinacni sistem "
            "koji izvodi konkretnu operaciju obrade uz elemente upravljanja, rada i kontrole."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Zasto se bojenje ili termicka obrada smatraju dijelom tehnoloskog procesa iako se oblik cesto ne mijenja?",
        "reference": (
            "Tehnoloski proces ne obuhvata samo promjenu oblika i dimenzija, vec i promjenu "
            "spoljasnjeg izgleda i unutrasnje strukture materijala. Zato operacije kao bojenje, "
            "termicka ili termohemijska obrada pripadaju tehnoloskom procesu i kad geometrija "
            "dijela ostane gotovo nepromijenjena."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Uporedi jednostavni, analiticki i sinteticki tehnoloski proces prema toku transformacije sirovina, dijelova i proizvoda.",
        "reference": (
            "Jednostavni proces ide pretezno jednom linijom od jedne sirovine ka proizvodu. "
            "Analiticki proces razlaze jednu sirovinu na vise izlaza, dok sinteticki proces "
            "spaja vise dijelova i podsklopova u jedinstven gotov proizvod."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Prema poglavlju 1.9, citiraj sustinu definicije: tehnologija je nauka koja proucava nacine izvodjenja prerada/obrada.",
        "reference": (
            "Tehnologija je nauka koja proucava nacine izvodjenja pojedinih prerada i obrada, "
            "odnosno postupke kojima se materijal i proizvod dovode do trazenog stanja i kvaliteta."
        ),
        "expects_retrieval": True,
    },
    {
        "question": "Dobar dan, mozes li mi preporuciti neki restoran u blizini?",
        "reference": (
            "Ovo je pitanje opste prirode koje ne zahtijeva pretragu u bazi tehnicke "
            "dokumentacije."
        ),
        "expects_retrieval": False,
    }
]