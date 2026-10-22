# Fish Tracking RT-DETR

Projekt je prototyp webové aplikace pro automatické rozpoznávání ryb na fotografii, počítání jejich počtu a odhad délky z obrazu podle referenční stupnice. 
Aplikace je postavená jako jednoduché demo v `Gradio` a běží nad modelem `RT-DETR` načteným z váhy `best.pt`.

## Co aplikace dělá

- nahraje fotografii s rybami a měřítkem,
- detekuje objekty tříd `fish` a `scale`,
- spočítá počet ryb,
- odhadne délku každé ryby v centimetrech,
- zobrazí výsledek přímo na obrázku,
- vyexportuje výsledky do CSV.

## Struktura projektu

- `app.py` - hlavní aplikace s UI, inferencí a výpočtem délky ryb
- `best.pt` - natrénované váhy modelu RT-DETR
- `requirements.txt` - Python závislosti

## Tech stack

### Frontend a UI

- `Gradio` pro rychlé webové rozhraní
- `gradio.Blocks` pro jednoduché rozvržení do dvou sloupců
- `gr.Image`, `gr.Number`, `gr.Slider`, `gr.Markdown`, `gr.File` pro nahrání dat, nastavení parametrů a zobrazení výstupu

### Model pro detekci

- `Ultralytics RT-DETR` jako object detection model
- lokální váhy `best.pt`
- inference přes `model.predict(...)`

### Výpočet a zpracování dat

- `NumPy` pro výpočet délky z bounding boxů
- `csv` a `io` pro generování CSV souboru
- `Pillow` pro práci s obrázky
- `OpenCV` v headless verzi jako běžná podpůrná závislost pro obrazové workflow

### Nasazení

- `Hugging Face Spaces`
- `spaces.GPU` pro běh inference na GPU

## Hugging Face, dataset a trénink

Aplikace je nasazená na **Hugging Face Spaces**, které slouží pro hostování ML demo aplikací. V tomto projektu Spaces umožňuje zpřístupnit model v prohlížeči bez vlastního serveru a s podporou GPU pro rychlejší inferenci.

Jádrem řešení je model **RT-DETR**, tedy moderní **object detection** architektura. Model na obrázku vyhledá objekty a vrátí jejich **bounding boxy**. V aplikaci pracuje se dvěma třídami:

- `fish`
- `scale`

RT-DETR je pro tento typ úlohy vhodný, protože umí najít více objektů na jednom snímku a dobře funguje i u různě natočených ryb nebo částečně překrytých objektů.

Model byl trénovaný na datasetu z **Roboflow**. Dataset obsahuje anotované fotografie ryb ve formátu vhodném pro detekční modely a je rozdělený na `train`, `val` a `test`. Roboflow při exportu provedl i základní předzpracování, například sjednocení orientace obrázků, resize a jednoduché augmentace.

Samotný trénink probíhal pomocí knihovny `Ultralytics`. Skript načte předtrénovaný model, například `rtdetr-l.pt`, a poté ho doladí na konkrétní dataset pomocí parametrů jako počet epoch, velikost obrázků, batch size, learning rate a device pro GPU. Výsledkem je soubor `best.pt`, který se následně používá v aplikaci pro detekci ryb na nových fotografiích.

## Proč jsou tyto technologie vhodné

### Gradio

Gradio je dobrá volba pro rychlý prototyp a akademické demo, protože:

- umožní sestavit UI velmi rychle,
- nevyžaduje velké množství boilerplate kódu,
- dobrá kompatibilita s Hugging Face Spaces,
- je vhodné pro interní ověření modelu i pro demonstraci zadavateli.

### RT-DETR

RT-DETR je moderní object detection architektura vhodná pro úlohy, kde je potřeba:

- najít více objektů na jedné fotografii,
- pracovat s různě natočenými objekty,
- mít detekci i s rozumnou přesností v jednom průchodu modelu.

V tomto projektu je výhodné, že model vrací bounding boxy pro ryby i pro měřítko a z nich lze přímo odvodit počet i orientační délku.

### Ultralytics

Ultralytics ekosystém je praktický, protože:

- zjednodušuje inferenci,
- má srozumitelné API,
- snadno se přenáší mezi lokálním během a cloudovým nasazením,
- dobře se hodí pro prototypy i menší produkční nástroje.

### CSV export

CSV je jednoduchý formát pro  analytické zpracování:

- otevře se v Excelu, R, Pythonu i MATLABu,
- je vhodný pro další statistickou analýzu,
- umožní přenos výsledků mezi týmem a zadavatelem.

## Vazba na zadání

Projekt odpovídá těmto bodům ze zadání:

- automatizace kvantitativní analýzy fotografií ryb,
- sečtení ryb na snímku,
- změření jejich velikosti,
- práce s referenčním měřítkem na fotografii,
- možnost dalšího rozvoje směrem k více druhům ryb,
- export výsledků pro další zpracování.

### Co je v současné verzi implementované

- detekce ryb a měřítka,
- výpočet délky ryb podle kalibrace z měřítka,
- zobrazení anotovaného obrázku,
- CSV export se seznamem a statistikou,
- jednoduché webové rozhraní pro nahrání obrázku.

## Jak to funguje technicky

1. Uživatel nahraje obrázek a zadá skutečnou délku měřítka v centimetrech.
2. Model `RT-DETR` detekuje objekty na snímku.
3. Z bounding boxů se oddělí detekce `fish` a `scale`.
4. Velikost měřítka v pixelech se převede na pixely na centimetr.
5. Délka ryby se dopočítá z diagonály jejího bounding boxu.
6. Výsledek se zobrazí na obrázku, v textu a uloží do CSV.

## Limity aktuální verze

- měření je odvozené z bounding boxu, ne z přesného obrysu těla ryby,
- kvalita výsledku závisí na správném umístění měřítka v obraze,
- při více měřítkách se bere největší detekce,
- aplikace zatím pracuje hlavně jako demonstrační nástroj, ne jako plnohodnotný datový systém.

## Poznámka k modelu

V kódu se používají dvě třídy:

- `fish`
- `scale`

To znamená, že aktuální verze je připravená na detekci ryb a referenční stupnice
