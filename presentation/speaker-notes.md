# Speaker notes - Deepfake Detection

Πλήρες ελληνικό script για παρουσίαση από laptop. Τα αριθμητικά αποτελέσματα παρακάτω ακολουθούν τα committed evaluation CSVs. Οι διατυπώσεις για τα caveats ακολουθούν το final interactive presentation και το `content-audit.md`, ακόμη όπου είναι στενότερες από τις αρχικές διατυπώσεις του report.

## Γρήγορος οδηγός χειρισμού

- [SHOW] σημαίνει ότι αφήνω το συγκεκριμένο view ορατό και μιλάω χωρίς να αλλάζω οθόνη.
- [CLICK] σημαίνει συγκεκριμένη ενέργεια στο interface.
- [PAUSE] σημαίνει σύντομη παύση για να διαβαστεί το οπτικό στοιχείο ή να ακουστεί το συμπέρασμα.
- Τα βέλη αριστερά και δεξιά αλλάζουν πράξη. Το `F` ή το κουμπί πλήρους οθόνης ενεργοποιεί fullscreen.
- Στις αρχιτεκτονικές χρησιμοποιώ τα tabs `Πώς λειτουργεί`, `Εκπαίδευση`, `Evidence`. Δεν χρειάζεται να ανοίξω κάθε stage button, εκτός αν ζητηθεί τεχνική λεπτομέρεια.
- Οι εικόνες μεγεθύνονται με click. Τις κλείνω πριν προχωρήσω, ώστε να μην εγκλωβιστεί η πλοήγηση σε dialog.
- Δεν λέω ότι τα επτά OOD subsets είναι επτά εντελώς unseen generator families. Λέω πάντα: **έξι νέα generator names και ένα ανεξάρτητα συλλεγμένο Midjourney subset**.
- Δεν λέω ότι το patch result αποδεικνύει αιτιωδώς την αξία του native local texture. Λέω ότι είναι το καλύτερο **μετρημένο** OOD αποτέλεσμα και χρειάζεται matched-resolution ablation.

## Αριθμητικό cheat sheet

| Μέγεθος ή metric | Ακριβής τιμή | Πώς το λέω |
|---|---:|---|
| Primary dataset μετά το cleaning | 59,882 | 59 χιλιάδες 882 εικόνες |
| Primary train | 43,127 | 43 χιλιάδες 127 |
| Primary validation | 4,792 | 4 χιλιάδες 792 |
| Primary ID test | 11,963 | 11 χιλιάδες 963 |
| DIRE ID test | 2,000 | subsample 2 χιλιάδων, όχι head-to-head |
| OOD dataset | 34,998 | 34 χιλιάδες 998 |
| OOD generator subsets | 7 | 6 νέα names και ανεξάρτητο Midjourney |
| Robustness subsample | 800 ανά level | μόνο 7 image-family pipelines |
| Best ID AUC | 0.997244 | ViT-LoRA, στην οθόνη 0.9972 |
| Best OOD accuracy | 0.677496 | patch-ensemble, στην οθόνη 0.6775 |
| Best robust mean | 0.922750 | ViT-LoRA |
| Best robust worst case | 0.727500 | ViT-LoRA |
| Strongest-level mean drop, noise | 0.350 | Gaussian noise έως sigma 0.15 |
| Strongest-level mean drop, downsampling | 0.221 | scale έως 0.25 |
| Strongest-level mean drop, JPEG | 0.003 | quality έως 60 |
| Strongest-level mean drop, blur | περίπου 0 | sigma έως 2.0 |

**Ορισμός gap που πρέπει να ειπωθεί ακριβώς:** `generalisation gap = ID accuracy - OOD accuracy`. Δεν είναι `ID AUC - OOD accuracy`.

---

# Ενότητα I - Ερευνητικό πρόβλημα

## Άνοιγμα

[SHOW το overview σε πλήρη οθόνη. Κοιτάζω πρώτα το ακροατήριο, όχι την οθόνη.]

**Ακριβές άνοιγμα:**

> Καλησπέρα σας. Θα ξεκινήσω με δύο αριθμούς που φαίνονται ασύμβατοι. Στο γνώριμο test set, ο καλύτερος detector μας φτάνει AUC 0.9972. Όταν όμως αλλάζουμε dataset και generator, ο καλύτερος σταματά σε accuracy 0.6775. Η απόσταση ανάμεσα σε αυτούς τους δύο αριθμούς είναι το πραγματικό αντικείμενο αυτής της εργασίας.

[PAUSE]

> Δεν εξετάσαμε ένα μόνο μοντέλο. Υλοποιήσαμε δέκα διαφορετικά deep-learning pipelines, από μικρά CNN μέχρι ViT με LoRA, CLIP, frequency και forensic detectors, patch-based multiple-instance learning και diffusion reconstruction. Το ερώτημα δεν ήταν απλώς ποιο έχει το καλύτερο score. Ήταν τι μαθαίνει το καθένα και αν αυτό που έμαθε επιβιώνει όταν αλλάξουν οι συνθήκες deployment.

[SHOW τα real/fake samples.]

> Οι επάνω εικόνες είναι πραγματικές και οι κάτω AI-generated. Δεν υπάρχει ένα οπτικό rule που να λύνει αξιόπιστα το πρόβλημα. Γι' αυτό η έξοδος κάθε pipeline είναι πιθανότητα `p(fake)` και όχι ένας χειροποίητος κανόνας.

> Η αρχική εικόνα μοιάζει με επιτυχία: τέσσερα pipelines ξεπερνούν το 0.99 ID AUC και το καλύτερο φτάνει 0.997244. Η ερευνητική ένταση εμφανίζεται μόνο όταν σφραγίσουμε ένα δεύτερο benchmark και ρωτήσουμε αν το detector αναγνωρίζει την έννοια της συνθετικής εικόνας ή απλώς το fingerprint των generators που ήδη ξέρει.

[CLICK διαδοχικά RQ1, RQ2, RQ3, RQ4, με μία πρόταση στο καθένα.]

> Στο RQ1 μετράμε τη cross-generator μεταφορά. Στο RQ2 τη σταθερότητα σε JPEG, blur, downsampling και noise. Στο RQ3 εξετάζουμε αν τα frequency artifacts είναι γενικά forensic cues ή dataset signatures. Και στο RQ4 χρησιμοποιούμε architecture-native diagnostics για να εντοπίζουμε shortcuts, όχι για να ισχυριστούμε ότι έχουμε αιτιώδη απόδειξη της απόφασης.

### Προαιρετικό σύντομο άνοιγμα

> Δέκα detectors φτάνουν έως 0.9972 AUC στο γνώριμο distribution, αλλά ο καλύτερος φτάνει μόνο 0.6775 accuracy στο held-out benchmark. Η παρουσίαση εξηγεί πώς ένα σχεδόν τέλειο score μπορεί να είναι επιστημονικά παραπλανητικό.

## Μετάβαση στην Ενότητα II

**Ακριβής πρόταση μετάβασης:**

> Για να κρίνουμε αν αυτή η πτώση ανήκει στα μοντέλα και όχι σε λάθος πειραματικό σχεδιασμό, πρέπει πρώτα να δούμε τα δεδομένα και τους εύκολους δρόμους που θα μπορούσαν να επιτρέψουν σε ένα detector να κλέψει.

[CLICK `Μεθοδολογία πειραματικού ελέγχου` ή το nav `2 - Δεδομένα`.]

---

# Ενότητα II - Δεδομένα και πειραματικός σχεδιασμός

## Τα δύο datasets και οι ρόλοι τους

[SHOW `ai-real-images`, topic `Samples`.]

> Το primary dataset είναι το `ai-real-images`. Μετά το cleaning κρατήσαμε 59,882 εικόνες, περίπου ισορροπημένες σε real και fake. Οι fake εικόνες προέρχονται από Stable Diffusion, Midjourney και DALL-E. Από το training portion δημιουργήσαμε stratified validation split 10%, με fixed seed 42. Το τελικό partition είναι 43,127 train, 4,792 validation και 11,963 untouched ID test εικόνες.

> Το validation χρησιμοποιείται για early stopping, Optuna objective και threshold selection. Το test χρησιμοποιείται μόνο για την τελική in-distribution αξιολόγηση.

[CLICK dataset `tiny-genimage`, άφησε `Samples`.]

> Το OOD benchmark είναι το `tiny-genimage`, με 34,998 εικόνες, ισορροπημένες ανά generator subset. Δεν χρησιμοποιείται σε training, hyperparameter tuning ή threshold selection. Περιλαμβάνει BigGAN, VQDM, SDv5, Wukong, ADM, GLIDE και Midjourney.

> Εδώ χρειάζεται ακριβής caveat. Δεν είναι σωστό να πούμε ότι και οι επτά generator families είναι πλήρως unseen, επειδή Midjourney υπάρχει και στο primary training mix. Το σωστό είναι: επτά held-out subsets, έξι νέα generator names και ένα ανεξάρτητα συλλεγμένο Midjourney subset. Άρα το protocol μετρά μαζί cross-generator, cross-dataset και content shift. Το Midjourney overlap πιθανότατα εξηγεί γιατί αυτό το subset είναι το ευκολότερο.

## Cleaning και leakage control

> Πριν από οποιοδήποτε training, ελέγξαμε corrupt ή truncated files, exact duplicates με SHA-1, near-duplicates με perceptual hash και train-test leakage. Από 60,000 primary images κρατήσαμε 59,882, άρα απορρίψαμε 118. Από 35,000 OOD images κρατήσαμε 34,998, άρα απορρίψαμε 2. Δεν διαγράψαμε σιωπηρά δεδομένα: το cleaning αποτυπώνεται σε manifest ώστε η απόφαση να παραμένει ελέγξιμη.

## Το resolution shortcut

[CLICK topic `Shortcut audit` στο primary dataset.]

> Αυτό είναι το σημαντικότερο data-quality εύρημα. Οι πραγματικές φωτογραφίες του primary dataset ήταν συστηματικά μεγαλύτερες και βαρύτερες από τις generated. Ένα CNN θα μπορούσε να μάθει ότι μεγάλη ή υψηλής λεπτομέρειας εικόνα σημαίνει real και ότι canonical διαστάσεις 256, 512 ή 1024 σημαίνουν fake. Θα πετύχαινε πολύ υψηλό ID score χωρίς να έχει μάθει deepfake detection.

> Για εννέα από τα δέκα pipelines, κάθε εικόνα περνά πρώτα από κοινό cache: resize της μικρής πλευράς σε 256 pixels και center crop σε 256 επί 256. Αυτό αφαιρεί original dimensions και aspect-ratio cues πριν το μοντέλο δει την εικόνα.

> Υπάρχει όμως μία ουσιαστική εξαίρεση. Το `patch-ensemble` παρακάμπτει αυτό το cache και διαβάζει native files. Αν η εικόνα είναι μικρότερη από patch 224 επί 224, γίνεται bilinear upscale για να χωρέσει. Επομένως original resolution, source resolution και upscale behavior παραμένουν πιθανοί confounders. Η υπεροχή του patch model δεν μπορεί να αποδοθεί μόνο στο local texture χωρίς matched-resolution ablation.

## EDA που οδήγησε στις υποθέσεις

[CLICK topic `Frequency`, πρώτα primary και μετά OOD.]

> Στο Fourier domain βλέπουμε διαφορά στη mid και high-frequency ενέργεια ανάμεσα σε real και generated εικόνες. Ο μηχανισμός είναι εύλογος: upsampling και synthesis stages αλλάζουν τις χωρικές συσχετίσεις των pixels και αφήνουν spectral traces. Όμως το σχήμα, ακόμη και το πρόσημο μέρους του spectral gap, δεν μένει σταθερό στο OOD dataset. Άρα το frequency signal υπάρχει, αλλά δεν είναι αυτομάτως generator-invariant.

[CLICK topic `Embedding`.]

> Στα frozen CLIP embeddings, μετά από PCA 50 και t-SNE σε δύο διαστάσεις, real και fake χωρίζονται αρκετά στο primary set, αλλά αναμειγνύονται περισσότερο στο OOD. Η t-SNE εδώ είναι exploratory projection, όχι metric απόδοσης. Μας έδωσε όμως μία καθαρή υπόθεση: ίσως ένα foundation embedding generalise καλύτερα από low-level fingerprints.

## Shared training contract

> Όλα τα pipelines καταλήγουν σε ένα logit, sigmoid και `p(fake)`. Τα 128-pixel custom και frequency CNNs χρησιμοποιούν dataset normalization. Τα pretrained backbones χρησιμοποιούν ImageNet normalization και το CLIP τη δική του. Η εκπαίδευση χρησιμοποιεί AdamW, cosine scheduling, warmup, early stopping στο validation AUC και light augmentation. Heavy blur, JPEG, rotation, cutout και mixup δεν μπήκαν στο training, επειδή θα κατέστρεφαν τα micro-artifacts που θέλαμε να μελετήσουμε. Επανέρχονται ως evaluation perturbations.

> Οκτώ pipelines tuned με Optuna, TPE sampler και MedianPruner. Τα `cnn-scratch` και `cnn-residual` έμειναν σκόπιμα untuned baselines. Αυτό είναι ιδιαίτερα σημαντικό για την ερμηνεία του residual αποτελέσματος.

## Μετάβαση στην Ενότητα III

**Ακριβής πρόταση μετάβασης:**

> Αφού αφαιρέσαμε τα προφανή shortcuts για εννέα pipelines και απομονώσαμε το OOD set από κάθε απόφαση training, μπορούμε τώρα να δούμε τα δέκα μοντέλα όχι ως κατάλογο αρχιτεκτονικών, αλλά ως δέκα επιστημονικές υποθέσεις για το πού βρίσκεται το transferable evidence.

[CLICK nav `3 - Αρχιτεκτονικές`.]

---

# Ενότητα III - Αρχιτεκτονικές και ερευνητικές υποθέσεις

## Εισαγωγή στην ενότητα

> Για κάθε pipeline θα δώσω έξι πράγματα: επιστημονικό ορισμό, μηχανισμό, rationale εκπαίδευσης, υπόθεση, μετρημένο αποτέλεσμα και περιορισμό. Τα ID scores δείχνουν αν το task είναι learnable. Το OOD score και το gap δείχνουν τι μεταφέρεται. Υπενθυμίζω ότι gap σημαίνει ID accuracy μείον OOD accuracy.

### 1. `cnn-scratch`

[CLICK `cnn-scratch`, tab `Πώς λειτουργεί`.]

> Επιστημονικά, είναι ένα μικρό convolutional neural network χωρίς pretrained prior. Η convolution μαθαίνει τοπικά φίλτρα με shared weights: τα πρώτα layers διαβάζουν ακμές και micro-textures και τα βαθύτερα συνδυάζουν μεγαλύτερα receptive fields.

> Ο μηχανισμός ξεκινά από RGB 128 επί 128. Το stride-1 stem δουλεύει στο πλήρες spatial lattice πριν από pooling, ώστε να μη χαθούν αμέσως high-frequency συσχετίσεις. Ακολουθούν τέσσερα Conv-BatchNorm-ReLU-MaxPool blocks, global average pooling, dropout και binary head. Έχει περίπου 0.98 εκατομμύρια parameters.

[CLICK tab `Εκπαίδευση`.]

> Εκπαιδεύεται με BCE και label smoothing, AdamW, cosine schedule, warmup και early stopping στο validation AUC. Δεν έγινε Optuna tuning. Ο λόγος ύπαρξής του είναι να αποτελεί ειλικρινές baseline και ταυτόχρονα end-to-end έλεγχο ότι data loaders, loss, metrics και evaluation harness λειτουργούν.

> Η υπόθεση είναι ότι ένα μικρό CNN μπορεί να μάθει τοπικά generator fingerprints χωρίς pretraining.

[CLICK tab `Evidence`.]

> Μετρημένο αποτέλεσμα: ID accuracy 0.901112, ID AUC 0.964753, OOD accuracy 0.548831 και gap 0.352280. Στο robustness sweep έχει mean perturbed accuracy 0.790062 και worst case 0.541250.

> Ο περιορισμός είναι ότι τα 128 pixels και η απουσία pretrained prior περιορίζουν την αναπαράσταση και τη μεταφορά. Το συμπέρασμα είναι: το primary task είναι εύκολο ακόμη και για μικρό CNN, αλλά το learned fingerprint μεταφέρεται ελάχιστα πάνω από chance.

### 2. `cnn-residual`

[CLICK `cnn-residual`, tab `Πώς λειτουργεί`.]

> Επιστημονικά, είναι pre-activation residual CNN με Squeeze-and-Excitation channel attention. Κάθε residual block μαθαίνει διόρθωση `F(x)` πάνω σε identity path, δηλαδή `y = x + F(x)`, ώστε τα gradients να περνούν ευκολότερα σε μεγαλύτερο βάθος. Το SE συνοψίζει κάθε feature channel και προβλέπει channel gates.

> Ο μηχανισμός κρατά το ίδιο RGB 128 επί 128 input με το baseline, για να απομονώσει την επίδραση της αρχιτεκτονικής. Χρησιμοποιεί full-resolution stem, τρία stages pre-activation residual blocks, transitions από 64 σε 128 και 256 channels, SE attention, global pooling και EMA-scored binary head. Έχει περίπου 2.8 εκατομμύρια parameters.

[CLICK `Εκπαίδευση`.]

> Εκπαιδεύεται έως 40 epochs με AdamW, cosine warmup και EMA decay 0.999. Το τελευταίο BatchNorm scale κάθε branch αρχικοποιείται στο μηδέν, ώστε το δίκτυο να ξεκινά κοντά στην identity function. Και αυτό το pipeline έμεινε untuned.

> Η υπόθεση είναι ότι μεγαλύτερο βάθος, residual gradient flow και channel attention θα ξεπεράσουν το απλό CNN.

[CLICK `Evidence`.]

> Το αποτέλεσμα δεν στηρίζει αυτή την υπόθεση στην παρούσα υλοποίηση: ID accuracy 0.786843, AUC 0.867183, OOD 0.518973 και gap 0.267870. Το mean perturbed είναι 0.731125 και το worst case 0.513750.

> Προσοχή στην ερμηνεία: το gap είναι αριθμητικά το μικρότερο, αλλά μόνο επειδή το ID ceiling είναι ήδη χαμηλό. Δεν είναι ένδειξη ισχυρής γενίκευσης. Η βασική limitation είναι η untuned optimization. Το αποτέλεσμα δεν απορρίπτει residual connections ή SE. Λέει ότι το συγκεκριμένο training recipe χρειάζεται την ίδια tuning μεταχείριση με τα υπόλοιπα pipelines.

### 3. `cnn-finetune`

[CLICK `cnn-finetune`, tab `Πώς λειτουργεί`.]

> Επιστημονικά, είναι transfer learning με ImageNet-pretrained EfficientNet-B0. Τα MBConv inverted bottlenecks, οι depthwise separable convolutions και τα SE gates παρέχουν parameter-efficient feature extraction. Το μοντέλο δεν μαθαίνει όραση από το μηδέν, αλλά προσαρμόζει ένα ώριμο feature space στο forensic boundary.

> Ο μηχανισμός παίρνει RGB 224 επί 224 με ImageNet normalization, περνά από EfficientNet-B0, global pooling, dropout και ένα single-logit head. Έχει περίπου 5 εκατομμύρια trainable parameters.

[CLICK `Εκπαίδευση`.]

> Η εκπαίδευση είναι two-stage. Για τρία epochs παγώνει ο backbone και μαθαίνει μόνο το νέο head. Μετά γίνεται unfreeze με discriminative learning rates: τα πρώιμα generic layers αλλάζουν πιο αργά και το head πιο γρήγορα. Έγιναν 12 Optuna trials και επιλέχθηκε focal loss με gamma περίπου 2.94.

> Η υπόθεση είναι ότι το ImageNet prior θα μειώσει το κόστος εκμάθησης χρήσιμων οπτικών features και θα ανεβάσει την ακρίβεια.

[CLICK `Evidence`.]

> Αυτό επιβεβαιώνεται in-distribution αλλά όχι στη μεταφορά. ID accuracy 0.955948, AUC 0.993045, OOD 0.563632 και gap 0.392315, το μεγαλύτερο του project. Στο robustness έχει mean 0.859125 και worst 0.497500.

> Η limitation είναι έντονη specialization: το powerful prior προσαρμόστηκε πολύ καλά στα primary fingerprints. Το full fine-tuning δίνει εξαιρετικό fit, όχι εγγύηση cross-generator transfer.

### 4. `vit-lora`

[CLICK `vit-lora`, tab `Πώς λειτουργεί`.]

> Επιστημονικά, είναι Vision Transformer με parameter-efficient fine-tuning μέσω LoRA. Η εικόνα χωρίζεται σε 196 patches των 16 επί 16 pixels. Η self-attention επιτρέπει σε κάθε patch να αλληλεπιδρά με όλα τα υπόλοιπα και το CLS token συγκεντρώνει global evidence. Η LoRA κρατά παγωμένο το μεγάλο weight matrix και μαθαίνει low-rank update `B επί A` στα query, key και value projections.

> Ο μηχανισμός χρησιμοποιεί RGB 224 επί 224, patch embeddings, positional embeddings, frozen ViT-Base blocks, LoRA adapters και classifier πάνω στο CLS token. Έχει 86 εκατομμύρια total parameters, αλλά περίπου 1.2 εκατομμύρια trainable.

[CLICK `Εκπαίδευση`.]

> Εκπαιδεύονται μόνο οι LoRA adapters και το head. Σε 12 Optuna trials επιλέχθηκε rank 32, το μεγαλύτερο διαθέσιμο, και focal loss. Στο inference οι adapters μπορούν να συγχωνευθούν στα βασικά weights, χωρίς πρόσθετο latency από ξεχωριστό branch.

> Η υπόθεση είναι ότι global self-attention και περιορισμένη low-rank προσαρμογή θα δώσουν ισχυρό fit με λιγότερο overfitting από full fine-tuning.

[CLICK `Evidence`.]

> Είναι ο ID νικητής: accuracy 0.978183, AUC 0.997244 και Brier 0.017229, το χαμηλότερο του πίνακα. Στο OOD είναι δεύτερο με 0.602177 και gap 0.376005. Στο robustness είναι καθαρά πρώτο: mean perturbed 0.922750 και worst case 0.727500.

> Η limitation είναι ότι το global resized view εξακολουθεί να χάνει native-resolution detail. Η LoRA μειώνει την έκταση προσαρμογής, αλλά δεν εξαφανίζει το generator overfit.

### 5. `clip-probe`

[CLICK `clip-probe`, tab `Πώς λειτουργεί`.]

> Επιστημονικά, είναι frozen foundation embedding probe. Το CLIP ViT-B/32 έχει εκπαιδευτεί contrastively, ώστε matching image-text pairs να βρίσκονται κοντά σε κοινό embedding space. Εδώ ο image encoder μένει παγωμένος και μόνο ένα MLP μαθαίνει boundary πάνω σε normalized 512-dimensional embeddings.

> Ο μηχανισμός είναι RGB 224 επί 224, CLIP-specific normalization, frozen encoder, cached 512-D vector, MLP με ReLU και dropout και single binary logit. Το trainable μέρος είναι κάτω από ένα εκατομμύριο parameters.

[CLICK `Εκπαίδευση`.]

> Επειδή ο encoder είναι frozen, κάθε embedding υπολογίζεται μία φορά και γίνεται cache. Αυτό επέτρεψε 80 Optuna trials, από τα οποία 43 pruned, με focal loss στο winning configuration. Το backbone δεν μπορεί να overfit μέσω weight updates, αλλά το head μπορεί να μάθει dataset-specific semantic correlations.

> Η προεγγεγραμμένη υπόθεση ήταν ότι το semantic manifold του CLIP θα εντοπίζει απόκλιση από real photographs και θα γενικεύει καλύτερα από low-level detectors.

[CLICK `Evidence`.]

> Η υπόθεση δεν επιβεβαιώθηκε. ID accuracy 0.959208, AUC 0.993024, OOD 0.583662, τρίτη θέση, και gap 0.375546. Δεν περιλαμβάνεται στο κοινό robustness sweep.

> Η limitation είναι ότι το single 512-D vector απορρίπτει spatially local και πολύ υψηλής συχνότητας evidence. Επίσης δεν υπάρχει faithful spatial Grad-CAM από το probe. Η t-SNE δείχνει τη δομή του embedding space, όχι αιτιώδη explanation για μεμονωμένη απόφαση.

### 6. `two-stream`

[CLICK `two-stream`, tab `Πώς λειτουργεί`.]

> Επιστημονικά, είναι dual-representation fusion. Ο spatial branch αναλύει RGB content και texture. Ο δεύτερος branch αναλύει το logarithmic magnitude του 2-D FFT της luminance, όπου periodic generation traces εμφανίζονται πιο άμεσα.

> Ο μηχανισμός παίρνει την ίδια augmented εικόνα σε RGB 128 επί 128, δημιουργεί spatial features και frequency features, τα κάνει concatenation και τα περνά σε fusion head. Κάθε branch έχει επίσης auxiliary logit. Έχει περίπου 0.78 εκατομμύρια parameters.

[CLICK `Εκπαίδευση`.]

> Το FFT υπολογίζεται on-the-fly μετά το augmentation, ώστε οι δύο branches να βλέπουν ακριβώς το ίδιο sample. Το joint loss είναι fused loss συν auxiliary losses, με tuned weight περίπου 0.21. Έγιναν 20 Optuna trials και επιλέχθηκε focal loss.

> Η υπόθεση είναι ότι το spectral view παρέχει complementary evidence και ότι η fusion θα ξεπεράσει κάθε stream μόνο του.

[CLICK `Evidence`.]

> Η fusion πράγματι υπερτερεί από τα μεμονωμένα branches, αλλά όχι αρκετά στο OOD. ID accuracy 0.897685, AUC 0.960944, OOD 0.526402 και gap 0.371283. Robustness mean 0.786875, worst 0.496250.

> Η limitation είναι ότι ο frequency branch είναι ασθενής μόνος του και πολύ ευαίσθητος σε additive noise. Η fixed concatenation δεν μπορεί να απορρίπτει δυναμικά ένα αναξιόπιστο stream ανά εικόνα.

### 7. `freqcross`

[CLICK `freqcross`, tab `Πώς λειτουργεί`.]

> Επιστημονικά, είναι three-branch attention fusion. Επεκτείνει το two-stream με radial power profile: azimuthal averaging του Fourier spectrum σε bins ακτίνας, ώστε ένα MLP να διαβάζει τη συνολική spectral fall-off με σχετική rotation invariance.

> Ο μηχανισμός έχει spatial RGB branch, 2-D log-FFT branch και radial-profile branch. Ένα learned softmax gate προβλέπει ανά εικόνα weights που αθροίζουν σε ένα και σχηματίζει weighted feature sum. Έχει περίπου 0.82 εκατομμύρια parameters.

[CLICK `Εκπαίδευση`.]

> Οι τρεις branches, τα auxiliary heads και το gate εκπαιδεύονται end-to-end. Σε 20 Optuna trials κέρδισε BCE, 48 radial bins και feature width 256. Το coarse radial profile υποδεικνύει ότι το χρήσιμο signal βρίσκεται στη γενική spectral fall-off και όχι σε πολύ λεπτή binning.

> Η υπόθεση είναι ότι το adaptive gate θα επιλέγει το πιο αξιόπιστο cue ανά εικόνα.

[CLICK `Evidence`.]

> Είναι το καλύτερο 128-pixel pipeline σε ID AUC: accuracy 0.900276, AUC 0.965120, OOD 0.552632 και gap 0.347644. Τα mean fusion weights είναι περίπου 0.61 spatial, 0.29 radial και 0.10 FFT. Robustness mean 0.786000 και worst 0.505000.

> Η limitation είναι ότι attention πάνω σε branches αυξάνει τη διαγνωστική διαφάνεια, αλλά δεν διορθώνει distribution shift όταν αλλάζει το ίδιο το spectral fingerprint. Με Gaussian noise το frequency evidence πλημμυρίζει από τυχαία ενέργεια.

### 8. `srm-noise`

[CLICK `srm-noise`, tab `Πώς λειτουργεί`.]

> Επιστημονικά, είναι forensic residual detector. Τα fixed Spatial Rich Model high-pass kernels αφαιρούν μεγάλο μέρος του semantic content. Η learnable Bayar convolution παραμένει residual operator επειδή το κεντρικό coefficient είναι μείον ένα και το άθροισμα των γειτονικών coefficients είναι συν ένα.

> Ο μηχανισμός παίρνει RGB 128 επί 128, παράγει τρία fixed SRM residual maps και learnable constrained Bayar maps, τα ενώνει σε multi-channel tensor και χρησιμοποιεί μικρό CNN για την ταξινόμηση. Έχει περίπου 0.39 εκατομμύρια parameters.

[CLICK `Εκπαίδευση`.]

> Τα SRM filters δεν εκπαιδεύονται. Τα Bayar weights επανακανονικοποιούνται σε κάθε forward pass ώστε να διατηρούν τους constraints. Μόνο ο Bayar branch και το downstream CNN παίρνουν gradients. Έγιναν 24 Optuna trials και κέρδισε BCE.

> Η υπόθεση είναι ότι, αν αφαιρέσουμε το semantic content, το noise residual θα αποκαλύψει generator fingerprint που μεταφέρεται καλύτερα.

[CLICK `Evidence`.]

> Το residual είναι learnable αλλά όχι αρκετά transferable: ID accuracy 0.882387, AUC 0.951785, OOD 0.523116 και gap 0.359272. Robustness mean 0.760125 και worst 0.505000.

> Η limitation είναι διπλή. Αφαιρώντας semantics πετάμε και χρήσιμο evidence, και το λεπτό residual pattern καλύπτεται εύκολα από Gaussian noise. Το αποτέλεσμα δείχνει generator signal στο noise domain, όχι universal forensic signature.

### 9. `patch-ensemble`

[CLICK `patch-ensemble`, tab `Πώς λειτουργεί`.]

> Επιστημονικά, είναι multiple-instance learning. Κάθε εικόνα είναι ένα bag από patches και το label υπάρχει μόνο στο επίπεδο της εικόνας. Ένας shared EfficientNet-B0 encoder παράγει feature για κάθε patch. Η gated attention συνδυάζει tanh και sigmoid projections, κανονικοποιεί weights με softmax και σχηματίζει weighted sum, ώστε ένα artifact-bearing patch να μπορεί να κυριαρχήσει.

> Ο μηχανισμός παρακάμπτει το κοινό 256-square cache. Στο train παίρνει έξι random native-resolution crops 224 επί 224, τα περνά batched από shared encoder και κάνει gated MIL pooling πριν από το binary head. Έχει περίπου 4.7 εκατομμύρια parameters.

[CLICK `Εκπαίδευση`.]

> Κάθε image step απαιτεί πολλαπλά backbone passes, γι' αυτό έγιναν 8 ακριβά Optuna trials. Κέρδισαν focal loss με gamma περίπου 1.399, `K=6` και MIL hidden width 256.

> Η υπόθεση είναι ότι το transferable evidence βρίσκεται σε native-resolution local texture και αραιώνεται από global resize ή global pooling.

[CLICK `Evidence`.]

> Είναι το καλύτερο μετρημένο OOD pipeline: ID accuracy 0.968068, AUC 0.996283, OOD accuracy 0.677496 και gap 0.290572. Είναι 0.075319, δηλαδή περίπου 7.5 percentage points, πάνω από το ViT-LoRA στο OOD. Δεν περιλαμβάνεται στο robustness sweep.

> Τώρα η κρίσιμη limitation. Πρώτον, παρακάμπτει το shortcut-neutralised cache, άρα native resolution, class-correlated source resolution και bilinear upscaling μικρών αρχείων παραμένουν confounders. Δεύτερον, υπάρχει train-eval mismatch: το tuned training χρησιμοποιεί έξι random crops, αλλά ο current evaluation loader για `K=6` επιστρέφει μόνο τέσσερα deterministic corner crops. Center crop προστίθεται μόνο όταν `K=5`. Το published score διατηρείται επειδή η αλλαγή του loader θα ήταν νέο πείραμα, όχι διόρθωση παρουσίασης.

> Άρα το ασφαλές συμπέρασμα είναι: το native-patch MIL έχει το καλύτερο μετρημένο OOD αποτέλεσμα και στηρίζει την υπόθεση του local detail, αλλά δεν την αποδεικνύει αιτιωδώς. Χρειάζεται rerun με ακριβώς έξι eval crops και matched-resolution comparison μεταξύ native patches και patches από το κοινό 256-square cache.

### 10. `dire-recon`

[CLICK `dire-recon`, tab `Πώς λειτουργεί`.]

> Επιστημονικά, είναι reconstruction-based detection. Η γεωμετρική υπόθεση είναι ότι μια diffusion-generated εικόνα βρίσκεται πιο κοντά στο manifold ενός diffusion model από μια πραγματική φωτογραφία και επομένως ανακατασκευάζεται με διαφορετικό error pattern.

> Ο μηχανισμός χρησιμοποιεί DDIM inversion 20 deterministic steps για να χαρτογραφήσει την εικόνα σε latent noise. Με Stable Diffusion v1.5 κάνει reverse reconstruction, υπολογίζει το absolute error map `|x - x-hat|`, το κάνει cache σε 256 επί 256 και εκπαιδεύει EfficientNet-B0 classifier πάνω στο error map, όχι πάνω στο original RGB. Ο classifier έχει περίπου 5 εκατομμύρια parameters.

[CLICK `Εκπαίδευση`.]

> Το ακριβό diffusion reconstruction γίνεται μία φορά ανά εικόνα και αποθηκεύεται. Μετά το Optuna search και το training είναι συμβατικό EfficientNet fine-tuning πάνω στα DIRE maps. Έγιναν 20 trials και κέρδισε BCE.

> Η υπόθεση είναι ότι synthetic εικόνες θα ανακατασκευάζονται πιο πιστά από το SD v1.5 manifold από ό,τι πραγματικές φωτογραφίες.

[CLICK `Evidence`.]

> Το reconstruction error είναι learnable: ID accuracy 0.873000 και AUC 0.939858. Το OOD accuracy είναι 0.541500 και το gap 0.331500. Δεν περιλαμβάνεται στο robustness sweep.

> Η κρίσιμη limitation είναι ότι τα ID metrics προέρχονται από subsample 2,000 εικόνων αντί για το κοινό test `n=11,963`. Άρα είναι indicative, όχι head-to-head. Επιπλέον, το signal είναι δεμένο στο manifold του Stable Diffusion v1.5 και η πλήρης inference διαδρομή απαιτεί diffusion reconstruction για κάθε νέο input.

## Σύντομη σύνοψη αρχιτεκτονικών, αν χρειαστεί περικοπή

> Το plain CNN έδειξε ότι το ID task είναι εύκολο. Το untuned residual απέτυχε στην optimization. Το full fine-tuning πέτυχε πολύ υψηλό fit αλλά το μεγαλύτερο gap. Το ViT-LoRA ήταν πρώτο σε ID και robustness. Το CLIP δεν επιβεβαίωσε την υπόθεση καλύτερης OOD μεταφοράς. Τα frequency και residual models βρήκαν πραγματικό αλλά brittle signal. Το patch MIL είχε το καλύτερο μετρημένο OOD, με σοβαρά resolution και crop caveats. Το DIRE έδειξε learnable reconstruction error, αλλά μόνο σε ID subsample 2,000 και με model-specific manifold.

## Μετάβαση στην Ενότητα IV

**Ακριβής πρόταση μετάβασης:**

> Οι δέκα υποθέσεις χρησιμοποιούν διαφορετικά inputs και inductive biases, αλλά η συγκριτική αξιολόγηση πρέπει να βασίζεται σε κοινό protocol. Ας δούμε λοιπόν τι ακριβώς μετρήσαμε, σε ποιο sample και με ποιους περιορισμούς.

[CLICK nav `4 - Αξιολόγηση`.]

---

# Ενότητα IV - Πρωτόκολλα αξιολόγησης

## P1 - In-distribution

[CLICK `P1 In-distribution`.]

> Το πρώτο protocol απαντά μόνο αν το task είναι learnable σε matched conditions. Για εννέα pipelines χρησιμοποιεί το untouched primary test set των 11,963 εικόνων. Το DIRE χρησιμοποιεί 2,000. Αναφέρουμε accuracy, macro-F1, AUC-ROC, PR-AUC, precision, recall, MCC και Brier, καθώς και default και validation-tuned operating points.

> Το AUC είναι threshold-free ranking metric: πόσο συχνά ένα τυχαίο fake παίρνει μεγαλύτερο score από ένα τυχαίο real. Το Brier μετρά calibration error. Το P1 είναι αναγκαίο sanity check, όχι deployment conclusion.

## P2 - Cross-generator OOD

[CLICK `P2 Cross-generator OOD`.]

> Το δεύτερο protocol είναι το κεντρικό. Κάθε trained pipeline εφαρμόζεται untouched στις 34,998 εικόνες του `tiny-genimage`. Δεν γίνεται retraining, tuning ή threshold selection πάνω στο OOD set. Η class balance κάνει το περίπου 0.5 random baseline ουσιαστικό.

> Μετράμε overall OOD accuracy, per-generator accuracy και `gap = ID accuracy - OOD accuracy`. Επαναλαμβάνω την caveat: τα επτά subsets είναι held out, αλλά μόνο έξι generator names είναι νέα. Το Midjourney επανεμφανίζεται από ανεξάρτητη συλλογή. Επομένως μιλάμε για cross-dataset και cross-generator shift, όχι για απολύτως disjoint seven-family test.

## P3 - Robustness

[CLICK `P3 Robustness`.]

> Το robustness sweep χρησιμοποιεί subsample 800 test εικόνων ανά perturbation level. Εφαρμόζει τέσσερις perturbations σε πέντε levels: JPEG quality 100 έως 60, Gaussian blur sigma 0 έως 2, downsample scale 1 έως 0.25 και Gaussian noise standard deviation 0 έως 0.15.

> Η scope είναι αυστηρά περιορισμένη στα επτά image-family pipelines: `cnn-scratch`, `cnn-residual`, `cnn-finetune`, `vit-lora`, `two-stream`, `freqcross` και `srm-noise`. Τα `clip-probe`, `patch-ensemble` και `dire-recon` εξαιρούνται, επειδή καταναλώνουν specialised embeddings, patch bags ή reconstruction maps. Άρα δεν λέμε ότι ελέγξαμε robustness και των δέκα.

> Επίσης δεν γενικεύουμε πέρα από αυτές τις τέσσερις αλλοιώσεις και τα συγκεκριμένα strengths. Το sweep δείχνει out-of-the-box robustness, επειδή οι perturbations δεν χρησιμοποιήθηκαν ως training augmentation. Δεν αποτελεί γενική εγγύηση έναντι arbitrary transformations ή adversarial attacks.

## XAI - Architecture-native diagnostics

[CLICK `XAI Explainability`.]

> Δεν επιβάλαμε μία explanation method σε όλα τα models. Για CNNs χρησιμοποιήσαμε Grad-CAM, για ViT attention rollout, για patch MIL τα πραγματικά attention weights, για `freqcross` branch weights, για SRM τα residual maps, για DIRE το ίδιο το reconstruction-error input και για CLIP τη δομή του embedding space.

> Αυτές οι μέθοδοι έχουν διαφορετική faithfulness. Τα MIL weights και τα actual input maps συνδέονται άμεσα με τον μηχανισμό. Το Grad-CAM και το rollout είναι post-hoc ή attention-based diagnostics. Τα χρησιμοποιούμε για shortcut audit και qualitative inspection. Δεν λέμε ότι αποδεικνύουν αιτιωδώς γιατί πάρθηκε μία απόφαση.

## Μετάβαση στην Ενότητα V

**Ακριβής πρόταση μετάβασης:**

> Με το protocol πλέον σαφές, μπορούμε να διαβάσουμε τα αποτελέσματα χωρίς να μπερδεύουμε την ικανότητα fitting με την ικανότητα μεταφοράς. Και εκεί η αρχική leaderboard ανατρέπεται.

[CLICK nav `5 - Αποτελέσματα`.]

---

# Ενότητα V - Συγκριτικά αποτελέσματα

## 1. ID performance

[CLICK tab `ID performance`.]

> Στο in-distribution test, τέσσερα 224-pixel pipelines ξεπερνούν 0.99 AUC: ViT-LoRA 0.997244, patch-ensemble 0.996283, CNN fine-tune 0.993045 και CLIP probe 0.993024. Το ViT-LoRA έχει επίσης την καλύτερη accuracy, 0.978183, και το χαμηλότερο Brier, 0.017229.

> Αυτό δεν σημαίνει ότι η πιο σύνθετη αρχιτεκτονική κέρδισε επιστημονικά. Η ID κατάταξη εξηγείται σε μεγάλο βαθμό από input resolution επί prior strength. Τα 224-pixel pretrained models και το native patch model βλέπουν πλουσιότερο detail από τα 128-pixel custom CNNs.

> Το DIRE row είναι με αστερίσκο: `n=2,000`, όχι 11,963. Δεν κάνω άμεση head-to-head σύγκριση του ID score του με τα άλλα εννέα.

## 2. OOD vs random

[CLICK tab `OOD vs random`.]

> Όταν αλλάζει το distribution, το patch-ensemble περνά πρώτο με OOD accuracy 0.677496. Το ViT-LoRA είναι δεύτερο με 0.602177, το CLIP τρίτο με 0.583662 και το fine-tuned EfficientNet τέταρτο με 0.563632. Το καλύτερο model βρίσκεται μόνο 0.177296 πάνω από το empirical random baseline του committed comparison.

> Το σημαντικό δεν είναι μόνο ότι άλλαξε ο νικητής. Είναι η κλίμακα της πτώσης. Για τα competitive models, τα gaps εκτείνονται από 0.290572 έως 0.392315, δηλαδή περίπου 29.1 έως 39.2 accuracy points. Το residual έχει μικρότερο αριθμητικό gap 0.267870, αλλά ξεκινά από ID accuracy μόλις 0.786843 και φτάνει OOD 0.518973. Γι' αυτό το patch έχει το μικρότερο meaningful gap ανάμεσα στους competitive detectors, όχι το αριθμητικά μικρότερο gap συνολικά.

[SHOW ή διάβασε μόνο αν ζητηθούν όλοι οι αριθμοί.]

| Pipeline | ID accuracy | ID AUC | OOD accuracy | Gap |
|---|---:|---:|---:|---:|
| patch-ensemble | 0.968068 | 0.996283 | 0.677496 | 0.290572 |
| vit-lora | 0.978183 | 0.997244 | 0.602177 | 0.376005 |
| clip-probe | 0.959208 | 0.993024 | 0.583662 | 0.375546 |
| cnn-finetune | 0.955948 | 0.993045 | 0.563632 | 0.392315 |
| freqcross | 0.900276 | 0.965120 | 0.552632 | 0.347644 |
| cnn-scratch | 0.901112 | 0.964753 | 0.548831 | 0.352280 |
| dire-recon* | 0.873000 | 0.939858 | 0.541500 | 0.331500 |
| two-stream | 0.897685 | 0.960944 | 0.526402 | 0.371283 |
| srm-noise | 0.882387 | 0.951785 | 0.523116 | 0.359272 |
| cnn-residual | 0.786843 | 0.867183 | 0.518973 | 0.267870 |

> Αστερίσκος: τα ID metrics του DIRE έχουν `n=2,000`.

## 3. Per-generator structure

[CLICK tab `Per generator`.]

> Οι στήλες της heatmap φωτίζονται και σκοτεινιάζουν μαζί. VQDM και BigGAN είναι γενικά τα δυσκολότερα, ενώ Midjourney το ευκολότερο. Αυτό δείχνει ότι η δυσκολία συνδέεται περισσότερο με distributional distance παρά με ένα model-specific blind spot.

> Όμως το Midjourney δεν είναι καθαρά unseen family. Υπάρχει στο training mix και επανεμφανίζεται ως ανεξάρτητη συλλογή. Επομένως δεν χρησιμοποιώ την ευκολία του ως απόδειξη universal transfer. Τη χρησιμοποιώ ως ένδειξη ότι generator similarity και dataset provenance επηρεάζουν έντονα τη μέτρηση.

## 4. Robustness

[CLICK tab `Robustness`.]

> Στα επτά pipelines που ελέγχθηκαν, το ViT-LoRA έχει mean perturbed accuracy 0.922750 και worst case 0.727500. Είναι το μόνο με worst case πάνω από 0.70.

> Η αρχική προσδοκία ήταν ότι JPEG και blur θα είναι οι μεγαλύτερες απειλές, επειδή αφαιρούν high-frequency detail. Τα δεδομένα έδειξαν το αντίθετο. Η μέση πτώση στο ισχυρότερο Gaussian noise είναι 0.350 και στο heavy downsampling 0.221. Στο JPEG quality 60 είναι μόλις 0.003 και στο blur sigma 2 περίπου μηδέν.

> Η ερμηνεία είναι cue-specific. Το additive noise δεν αφαιρεί απλώς high frequencies. Τα γεμίζει με τυχαία ενέργεια και καλύπτει τα λεπτά residuals. Γι' αυτό τα frequency-aware pipelines είναι ιδιαίτερα noise-fragile, ενώ το ViT, που χρησιμοποιεί περισσότερο global structure, κρατά καλύτερα.

> Η διατύπωση πρέπει να μείνει στενή: αυτό ισχύει για 800 εικόνες ανά level, επτά image-family pipelines, τέσσερις perturbations και τα συγκεκριμένα strengths. Δεν είναι robustness certificate.

## Σύνοψη των ευρημάτων σε τέσσερις προτάσεις

> Πρώτον, η υπόθεση ότι το CLIP θα generalise καλύτερα διαψεύστηκε. Δεύτερον, το frequency signal υπάρχει αλλά είναι brittle και generator-dependent. Τρίτον, το native-patch MIL έχει το καλύτερο μετρημένο OOD αποτέλεσμα, αλλά το resolution confound και το 6-train/4-eval crop mismatch εμποδίζουν αιτιώδες claim. Τέταρτον, το ID AUC δεν προβλέπει deployment readiness.

## Μετάβαση στην Ενότητα VI

**Ακριβής πρόταση μετάβασης:**

> Άρα η σωστή ερώτηση για deployment δεν είναι ποιο μοντέλο έχει ακόμη ένα δεκαδικό στο γνώριμο test set. Είναι ποιο σύστημα γνωρίζει τα όριά του, καλύπτει νέα distributions και διατηρεί αξιόπιστη πιθανότητα όταν αλλάζει ο κόσμος γύρω του.

[CLICK nav `6 - Συμπεράσματα`.]

---

# Ενότητα VI - Συμπεράσματα, περιορισμοί και μελλοντική εργασία

## Η εφαρμογή

[SHOW το runtime flow και το persisted inference.]

> Η εφαρμογή κάνει την κοινή inference σύμβαση χειροπιαστή. Ένα Streamlit frontend επικοινωνεί με FastAPI backend. Ο χρήστης επιλέγει pipeline, το backend το ζεσταίνει, δέχεται upload, προβλέπει, επιστρέφει explanation όπου έχει νόημα και αποθηκεύει input μαζί με uniform JSON response.

> Ο backend επαναχρησιμοποιεί τους ίδιους model builders και transforms με τα notebooks. Αυτό είναι correctness choice: αποφεύγουμε δεύτερη υλοποίηση που θα μπορούσε να αλλάξει resize, normalization ή layer wiring και να δίνει διαφορετικές προβλέψεις από την αξιολόγηση.

> Επειδή το σύστημα σχεδιάστηκε για μία consumer GPU, μόνο ένα pipeline είναι resident κάθε στιγμή. Η κατάσταση είναι `cold`, `warming` ή `warm`. Η επιλογή νέου pipeline κάνει eviction του προηγούμενου, καθαρίζει GPU references και cache, φορτώνει το νέο model και ένα lock σειριοποιεί το inference.

> Κάθε pipeline επιστρέφει `final.label`, `final.p_fake` και λίστα `components`. Ένα απλό CNN έχει ένα component. Το two-stream μπορεί να δείξει spatial και frequency sub-scores. Κάθε inference αποθηκεύεται με timestamp μαζί με την εικόνα και το JSON, για traceability.

> Το UI εκθέτει σήμερα έξι από τα δέκα pipelines. Τα `freqcross`, `srm-noise`, `patch-ensemble` και `dire-recon` είναι trained και documented, αλλά δεν έχουν ακόμη wired adapters στον selector. Επίσης το CLIP δεν προσφέρει spatial explanation, επειδή ο classifier βλέπει μόνο embedding.

## Limitations που λέγονται ρητά

> Πρώτη limitation: έχουμε ένα μόνο OOD dataset pairing. Το αποτέλεσμα είναι σταθερό σε δέκα αρχιτεκτονικές, αλλά χρειάζεται δεύτερο ανεξάρτητο benchmark.

> Δεύτερη: το OOD set έχει Midjourney overlap ως generator name. Άρα έχουμε έξι νέα names και ένα independently sourced overlap, όχι επτά πλήρως disjoint families.

> Τρίτη: το patch model έχει unresolved native-resolution και upscaling confound, επειδή δεν χρησιμοποιεί το κοινό 256-square cache. Επιπλέον εκπαιδεύεται με έξι random crops αλλά αξιολογείται με τέσσερα deterministic corners.

> Τέταρτη: το DIRE ID result βασίζεται σε 2,000 εικόνες και σε ένα Stable Diffusion v1.5 reconstruction manifold.

> Πέμπτη: τα δύο from-scratch baselines δεν tuned, άρα ειδικά το residual result είναι incomplete optimization experiment.

> Έκτη: το robustness sweep καλύπτει μόνο επτά image-family pipelines, sample 800 ανά level και συγκεκριμένες perturbations. Δεν καλύπτει CLIP, patch ή DIRE και δεν είναι general adversarial robustness claim.

> Έβδομη: τα thresholds και τα calibration metrics είναι in-distribution operating points. Δεν υπάρχει εγγύηση ότι παραμένουν σωστά μετά από distribution shift.

> Όγδοη: δεν αναφέρουμε bootstrap confidence intervals ή pairwise uncertainty για τις OOD διαφορές. Επομένως η κατάταξη είναι η μετρημένη κατάταξη αυτού του benchmark, όχι απόδειξη ότι κάθε μικρή διαφορά θα επαναληφθεί.

## Επόμενα βήματα, με σειρά επιστημονικής προτεραιότητας

> Πρώτο, rerun του patch evaluation με loader που επιστρέφει ακριβώς `K=6` deterministic crops και matched-resolution ablation: native patches εναντίον patches από το κοινό 256-square cache.

> Δεύτερο, δεύτερο πραγματικά generator-disjoint OOD dataset χωρίς Midjourney overlap, μαζί με bootstrap confidence intervals και pairwise comparisons.

> Τρίτο, διεύρυνση του training mix με GAN-era και older-diffusion families, επειδή η per-generator δυσκολία ακολουθεί distributional distance.

> Τέταρτο, Optuna tuning του `cnn-residual` και του `cnn-scratch`, ώστε το residual hypothesis να κριθεί δίκαια.

> Πέμπτο, noise-aware training ή denoising defenses, ειδικά για frequency και residual models.

> Έκτο, πλήρες DIRE πάνω στις 11,963 ID test εικόνες και reconstruction backbones πέρα από Stable Diffusion v1.5.

> Έβδομο, ensemble native local evidence με frequency-aware cue, αφού τα error patterns τους μπορεί να είναι συμπληρωματικά.

> Όγδοο, σύνδεση και των δέκα pipelines στο app, OOD recalibration, monitoring ανά source distribution και σαφής abstention ή human-review policy για αβέβαια samples.

## Κλείσιμο

[SHOW ξανά, αν χρειάζεται, τα δύο headline numbers ή μείνε στο deployment view. Κοίτα το ακροατήριο.]

**Ακριβές κλείσιμο:**

> Θα κλείσω με το βασικό συμπέρασμα. Για αυτό το primary dataset, το in-distribution deepfake detection μοιάζει σχεδόν λυμένο: AUC 0.9972. Για τον generator που το μοντέλο δεν έχει ακόμη δει, το πρόβλημα παραμένει ανοιχτό: καλύτερη μετρημένη accuracy 0.6775, και αυτή με σημαντικές validity caveats. Η ουσιαστική συνεισφορά της εργασίας δεν είναι άλλο ένα 0.99. Είναι ότι, σε δέκα διαφορετικές αρχιτεκτονικές και ένα κοινό protocol, δείχνει πόσο μακριά μπορεί να βρίσκεται ένα εντυπωσιακό laboratory score από την πραγματική δυνατότητα deployment.

[PAUSE]

> Σας ευχαριστώ. Είμαι έτοιμος για τις ερωτήσεις σας.

### Προαιρετικό σύντομο κλείσιμο

> Το 0.9972 αποδεικνύει ότι το γνώριμο task είναι learnable. Το 0.6775 αποδεικνύει ότι η μεταφορά παραμένει το ανοιχτό πρόβλημα. Για deployment, το OOD evidence και τα όρια του detector αξίζουν περισσότερο από ακόμη ένα ID decimal.

---

# Αναμενόμενες ερωτήσεις εξεταστών

## Πειραματικός σχεδιασμός

**Ερώτηση: Είναι πράγματι unseen και οι επτά OOD generators;**

> Όχι με την αυστηρή έννοια της generator family. Και τα επτά subsets είναι held out από training και tuning, αλλά το Midjourney name υπάρχει και στο primary mix. Άρα έχουμε έξι νέα generator names και ένα ανεξάρτητα συλλεγμένο Midjourney subset. Το αποτέλεσμα μετρά μαζί generator, dataset και content shift.

**Ερώτηση: Γιατί χρησιμοποιείτε accuracy στο OOD και AUC στο ID;**

> Το committed OOD comparison έχει ως headline overall accuracy και per-generator accuracy, σε class-balanced benchmark με explicit random baseline. Το ID table διατηρεί πλήρη metric battery, όπου το AUC απομονώνει ranking quality από το threshold. Δεν συγκρίνω AUC με accuracy μέσα στο gap: το gap είναι αυστηρά ID accuracy μείον OOD accuracy.

**Ερώτηση: Γιατί το random baseline δεν είναι ακριβώς 0.5;**

> Το CSV χρησιμοποιεί empirical dummy baseline περίπου 0.5002 στο συγκεκριμένο OOD label vector. Γι' αυτό το lift του patch είναι 0.177296 και όχι ακριβώς 0.177496. Στην προφορική παρουσίαση το στρογγυλεύω σε 0.500.

**Ερώτηση: Πώς αποτρέψατε leakage;**

> Με strict decode για corrupt files, SHA-1 για exact duplicates, DCT perceptual hash για near-duplicates και cross-split checks. Το primary test έμεινε untouched. Το OOD dataset δεν μπήκε σε training, Optuna ή threshold selection.

**Ερώτηση: Γιατί κοινό cache 256 επί 256, αλλά working sizes 128 και 224;**

> Το cache αφαιρεί original resolution και aspect-ratio shortcuts και επιταχύνει decoding. Μετά, κάθε pipeline κάνει deterministic ή train transform από το κοινό cache στη δική του working size. Τα custom CNNs δουλεύουν στα 128 για compute, τα pretrained backbones στα 224 επειδή εκεί αναμένουν input.

**Ερώτηση: Γιατί δεν χρησιμοποιήσατε heavy augmentation;**

> Επειδή blur, JPEG, aggressive resize, cutout ή mixup μπορούν να καταστρέψουν τα micro-texture και frequency cues που μελετάμε. Χρησιμοποιήσαμε light crop και horizontal flip. Τις καταστροφικές αλλοιώσεις τις κρατήσαμε για out-of-the-box robustness evaluation.

## Αρχιτεκτονικές και αποτελέσματα

**Ερώτηση: Γιατί το patch-ensemble κερδίζει;**

> Η plausible εξήγηση είναι ότι κρατά native local texture και η gated MIL attention επιλέγει artifact-bearing patches. Όμως δεν το αποδίδω αιτιωδώς μόνο σε αυτό, επειδή το model παρακάμπτει το 256-square cache, μπορεί να βλέπει resolution/upscale cues και έχει 6-train/4-eval crop mismatch. Χρειάζεται matched-resolution ablation και corrected rerun.

**Ερώτηση: Γιατί δεν διορθώσατε απλώς το patch eval loader;**

> Επειδή τότε θα παρήγαγα νέο αποτέλεσμα διαφορετικό από τα committed artifacts. Η παρουσίαση οφείλει να αναφέρει το published score και να αποκαλύπτει τον mismatch. Η διόρθωση είναι follow-up experiment με νέο evaluation output.

**Ερώτηση: Το μικρότερο gap δεν το έχει το residual;**

> Αριθμητικά ναι: 0.267870. Αλλά φτάνει μόνο 0.518973 OOD επειδή ξεκινά από χαμηλό ID accuracy 0.786843. Το patch έχει το μικρότερο meaningful gap ανάμεσα στα competitive detectors, 0.290572, και ταυτόχρονα το υψηλότερο OOD 0.677496.

**Ερώτηση: Γιατί απέτυχε το residual network;**

> Η ασφαλής ερμηνεία είναι optimization failure, όχι αρχιτεκτονική απόρριψη. Είναι βαθύτερο και πιο ευαίσθητο σε learning rate, warmup και EMA, αλλά δεν πήρε Optuna search. Χρειάζεται tuned rerun πριν κριθεί το residual design.

**Ερώτηση: Γιατί δεν κέρδισε το CLIP στο OOD;**

> Το CLIP κρατά high-level semantic και distributional structure, αλλά συμπιέζει την εικόνα σε 512-D global vector και απορρίπτει local high-frequency evidence. Το semantic signal μεταφέρεται αρκετά για τρίτη θέση, όχι όσο το native patch evidence σε αυτό το benchmark.

**Ερώτηση: Τι απέδειξαν τα frequency models;**

> Ότι το spectral signal είναι πραγματικό και complementary: η fusion ξεπερνά τους μεμονωμένους branches. Δεν απέδειξαν universal generator fingerprint. Το spectral gap αλλάζει στο OOD και τα models είναι ιδιαίτερα noise-fragile.

**Ερώτηση: Είναι το DIRE δίκαια συγκρίσιμο;**

> Όχι πλήρως. Το ID score του βασίζεται σε 2,000 εικόνες αντί για 11,963 και έχει ευρύτερη αβεβαιότητα. Επιπλέον το reconstruction prior είναι Stable Diffusion v1.5-specific. Το παρουσιάζουμε ως indicative experiment.

**Ερώτηση: Γιατί focal loss στα μεγάλα models και BCE στα μικρά;**

> Αυτό ήταν αποτέλεσμα Optuna, όχι κοινός προκαθορισμένος κανόνας. Focal κέρδισε στα high-capacity backbones και στο patch model, πιθανότατα επειδή εστιάζει σε δύσκολα boundary samples. BCE κέρδισε στα μικρότερα frequency/forensic networks και στο DIRE.

## Robustness και explainability

**Ερώτηση: Γιατί το Gaussian noise είναι χειρότερο από JPEG;**

> Το JPEG και το blur αφαιρούν μέρος των high frequencies, αλλά στα tested strengths αφήνουν αρκετό discriminative structure. Το additive noise πλημμυρίζει το ίδιο band με random energy και καλύπτει τα forensic residuals. Γι' αυτό πλήττει περισσότερο τα frequency-aware models.

**Ερώτηση: Γιατί εξαιρέθηκαν CLIP, patch και DIRE από robustness;**

> Επειδή το κοινό harness perturbs raw pixel inputs για image-family models. Για CLIP πρέπει να οριστεί perturbation πριν από embedding, για patch πριν από crop extraction και για DIRE πριν από reconstruction. Αυτές είναι specialised semantics και χρειάζονται ξεχωριστό protocol.

**Ερώτηση: Τα explainability maps αποδεικνύουν γιατί αποφάσισε το μοντέλο;**

> Όχι. Είναι architecture-native diagnostics με διαφορετική faithfulness. MIL weights και reconstruction/residual inputs είναι άμεσα δεμένα στον μηχανισμό. Grad-CAM, rollout και t-SNE είναι shortcut audits και qualitative evidence, όχι causal guarantees.

**Ερώτηση: Γιατί δεν δίνετε confidence intervals;**

> Τα committed evaluation artifacts δεν περιλαμβάνουν bootstrap confidence intervals, γι' αυτό δεν τα επινοούμε. Η σωστή συνέχεια είναι bootstrap OOD intervals και pairwise differences, ειδικά για μικρές αποστάσεις της leaderboard.

## Deployment

**Ερώτηση: Θα κάνατε deploy το patch-ensemble ως automatic truth detector;**

> Όχι. Το 0.677496 OOD accuracy και τα unresolved confounds δεν δικαιολογούν αυτόνομη κρίση. Θα το χρησιμοποιούσα ως risk signal σε human-review pipeline, με abstention, source-aware monitoring, OOD calibration και συνεχή ενημέρωση generator coverage.

**Ερώτηση: Γιατί μία GPU κρατά μόνο ένα model;**

> Για να χωρέσει το σύστημα σε consumer hardware και να αποφεύγει memory contention. Ο residency manager κάνει eviction, warm-up και serialized inference. Το trade-off είναι switching latency, όχι prediction inconsistency.

**Ερώτηση: Γιατί είναι wired μόνο έξι από τα δέκα pipelines;**

> Η εφαρμογή ολοκληρώνει το common inference contract για τους έξι core adapters. Τα τέσσερα extra architectures είναι trained και evaluated, αλλά χρειάζονται specialised adapters και explanation or preprocessing semantics. Η πλήρης σύνδεσή τους είναι δηλωμένο next step.

**Ερώτηση: Ποιο είναι το ένα βασικό συμπέρασμα;**

> Το υψηλό ID AUC αποδεικνύει ότι το detector λύνει το γνώριμο dataset. Δεν αποδεικνύει ότι αναγνωρίζει generated imagery γενικά. Η αξία deployment κρίνεται από cross-generator OOD evidence, calibration και robustness στο πραγματικό input distribution.
