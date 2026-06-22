(() => {
  const sections = [
    { key: 'overview', label: 'Ερευνητικό πρόβλημα' },
    { key: 'data', label: 'Δεδομένα και σχεδιασμός' },
    { key: 'architectures', label: 'Αρχιτεκτονικές' },
    { key: 'evaluation', label: 'Πρωτόκολλα αξιολόγησης' },
    { key: 'results', label: 'Συγκριτικά αποτελέσματα' },
    { key: 'deployment', label: 'Συμπεράσματα' }
  ];

  const researchQuestions = [
    {
      title: 'Κενό γενίκευσης',
      question: 'Μαθαίνει το μοντέλο την έννοια «synthetic image» ή το fingerprint των generators του training set;',
      conclusion: 'Τα ανταγωνιστικά pipelines χάνουν 29.1–39.2 accuracy points στο OOD benchmark. Το residual χάνει 26.8, αλλά μόνο επειδή ξεκινά από χαμηλό ID accuracy 0.7868. Το cross-generator test είναι το κεντρικό protocol.'
    },
    {
      title: 'Ανθεκτικότητα',
      question: 'Παραμένει η απόφαση σταθερή όταν η εικόνα συμπιεστεί, θολώσει, μικρύνει ή αποκτήσει θόρυβο;',
      conclusion: 'JPEG και blur έχουν σχεδόν μηδενικό κόστος, ενώ Gaussian noise και heavy downsampling καταστρέφουν το signal. Η ανθεκτικότητα είναι έντονα perturbation-specific.'
    },
    {
      title: 'Συχνοτικά χαρακτηριστικά',
      question: 'Τα φασματικά ίχνη είναι generator-invariant forensic evidence ή εύθραυστα dataset signatures;',
      conclusion: 'Η frequency fusion βοηθά, αλλά το spectral gap αλλάζει μεταξύ datasets και οι frequency-aware pipelines είναι οι πιο noise-fragile.'
    },
    {
      title: 'Ερμηνευσιμότητα',
      question: 'Πού και με ποιον μηχανισμό παράγεται το p(fake);',
      conclusion: 'Grad-CAM, attention rollout, MIL weights, residual maps και DIRE maps λειτουργούν ως shortcut audits και όχι μόνο ως οπτικά explanations.'
    }
  ];

  const datasetData = {
    primary: {
      title: 'ai-real-images',
      role: 'Training + in-distribution evaluation',
      size: '59,882 εικόνες',
      balance: '≈30k real / ≈30k fake',
      generators: 'Stable Diffusion · Midjourney · DALL·E',
      realSource: 'Pexels · Unsplash · WikiArt',
      samples: 'assets/samples-primary.png',
      spectrum: 'assets/spectrum-primary.png',
      embedding: 'assets/tsne-primary.png',
      properties: 'assets/properties-primary.png'
    },
    ood: {
      title: 'tiny-genimage',
      role: 'Held-out cross-dataset / cross-generator OOD probe',
      size: '34,998 εικόνες',
      balance: 'Balanced real/fake ανά generator',
      generators: 'BigGAN · VQDM · SDv5 · Wukong · ADM · GLIDE · Midjourney',
      realSource: '“Nature” photographs · ImageNet-class origin',
      samples: 'assets/samples-ood.png',
      spectrum: 'assets/spectrum-ood.png',
      embedding: 'assets/tsne-ood.png',
      properties: 'assets/properties-ood.png'
    }
  };

  const dataTopics = {
    samples: {
      title: 'Οπτική ετερογένεια και μετατόπιση περιεχομένου',
      caption: 'Αντιπροσωπευτικά samples του επιλεγμένου dataset.',
      analysis: 'Οι κλάσεις δεν διαχωρίζονται από ένα προφανές ανθρώπινο cue. Το OOD set είναι πιο object-centric και καλύπτει GAN-era έως νεότερες diffusion families, δημιουργώντας ταυτόχρονα generator shift και content shift.',
      bullets: ['Δεν χρησιμοποιούμε χαμηλής ανάλυσης benchmark.', 'Η OOD συλλογή δεν συμμετέχει σε tuning ή threshold selection.', 'Η ισορροπία κλάσεων κάνει το 0.5 ουσιαστικό random baseline.']
    },
    frequency: {
      title: 'Συχνοτικό αποτύπωμα της διαδικασίας παραγωγής',
      caption: 'Average 2-D Fourier log-magnitude και radial power evidence.',
      analysis: 'Η παραγωγική διαδικασία και τα upsampling stages μεταβάλλουν τις χωρικές συσχετίσεις των pixels. Στο Fourier domain αυτό εμφανίζεται ως δομημένη διαφορά mid/high-frequency ενέργειας. Το κρίσιμο εύρημα είναι ότι το πρόσημο και το σχήμα της διαφοράς δεν παραμένουν απολύτως σταθερά στο OOD set.',
      bullets: ['Το FFT δεν είναι classifier· είναι δεύτερη learned είσοδος.', 'Η radial averaging συμπυκνώνει το 2-D spectrum σε 1-D profile.', 'Η additive noise perturbation καταστρέφει άμεσα αυτό το cue.']
    },
    embedding: {
      title: 'Διαχωρισιμότητα στο embedding space του CLIP',
      caption: 'PCA(50) → t-SNE(2) των frozen CLIP embeddings.',
      analysis: 'Το CLIP έχει μάθει υψηλού επιπέδου semantics μέσω contrastive image-text pretraining. Στο primary set τα embeddings real/fake σχηματίζουν αρκετά διακριτές περιοχές, όμως στο OOD set αναμειγνύονται. Η εικόνα προαναγγέλλει ότι το semantic signal υπάρχει αλλά δεν είναι generator-invariant.',
      bullets: ['512-D embedding ανά εικόνα.', 'Frozen encoder: τα embeddings μπορούν να γίνουν cache.', 'Η t-SNE είναι exploratory projection, όχι performance metric.']
    },
    shortcut: {
      title: 'Έλεγχος συσχέτισης μεταξύ ανάλυσης και ετικέτας',
      caption: 'Διαστάσεις και file-size distributions ανά label.',
      analysis: 'Στο primary dataset οι πραγματικές φωτογραφίες ήταν συστηματικά μεγαλύτερες και βαρύτερες. Η κοινή 256² cache εξουδετερώνει αυτόν τον δρόμο για τα εννέα cache-based pipelines. Το patch-ensemble είναι σκόπιμη εξαίρεση: διαβάζει native files, άρα η υπεροχή του χρειάζεται matched-resolution ablation πριν αποδοθεί αιτιωδώς μόνο στο local texture.',
      bullets: ['Deduplication και cross-split leakage checks πριν από την αξιολόγηση.', '256² shortcut neutralisation για 9/10 pipelines.', 'Το native-patch αποτέλεσμα έχει unresolved resolution/upscale confound.']
    }
  };

  const architectures = [
    {
      key: 'cnn-scratch', name: 'cnn-scratch', family: 'From-scratch CNN', subtitle: 'Μικρό convolutional baseline με full-resolution stem και χωρίς pretrained prior.',
      hypothesis: 'Ένα μικρό CNN μπορεί να μάθει τοπικά generative fingerprints χωρίς pretraining και να ορίσει ένα ειλικρινές baseline.',
      diagram: 'assets/architectures/cnn-scratch.png', evidence: 'assets/evidence/cnn-scratch.png', evidenceLabel: 'Grad-CAM στο τελευταίο convolutional block.',
      science: 'Ένα CNN μαθαίνει τοπικά φίλτρα που μοιράζονται βάρη σε όλη την εικόνα. Τα πρώτα layers ανιχνεύουν ακμές και micro-textures, ενώ τα βαθύτερα συνθέτουν μεγαλύτερα receptive fields. Εδώ το stride-1 stem αποφεύγει early pooling ώστε οι λεπτές, υψηλής συχνότητας συσχετίσεις να παρατηρηθούν πριν από κάθε χωρική συμπίεση.',
      formula: 'hₗ = Pool(ReLU(BN(Wₗ * hₗ₋₁)))  →  p(fake) = σ(wᵀ·GAP(h₄))',
      stages: [
        ['RGB 128²', 'Dataset-normalized RGB. Η μικρότερη ανάλυση μειώνει compute αλλά περιορίζει το finest forensic detail.'],
        ['Full-res stem', 'Conv 3×3, stride 1 χωρίς pooling. Το δίκτυο διαβάζει πρώτα το πλήρες spatial lattice.'],
        ['4 conv blocks', 'Conv–BN–ReLU–MaxPool, με αύξηση channels 64→128→256 και σταδιακή αύξηση receptive field.'],
        ['Global pooling', 'Το GAP μετατρέπει κάθε feature map σε μία τιμή και κάνει την απόφαση σχετικά position-invariant.'],
        ['Binary head', 'Dropout και Linear(256→1) παράγουν logit, το οποίο μετατρέπεται σε calibrated p(fake) μέσω sigmoid.']
      ],
      specs: { Input: 'RGB 128×128', Normalization: 'Dataset mean/std', Parameters: '≈0.98M', Search: 'Untuned baseline', Loss: 'BCE + label smoothing', Explainability: 'Grad-CAM' },
      training: 'AdamW με cosine schedule, 3-epoch warmup και early stopping στο validation AUC. Η light augmentation διατηρεί τα micro-artifacts αντί να τα καταστρέφει με blur/JPEG. Η απλότητα του μοντέλου το κάνει χρήσιμο ως ελεγκτή όλου του evaluation harness.',
      limitation: 'Δεν έγινε Optuna tuning. Η ανάλυση 128² και η απουσία pretrained prior περιορίζουν τόσο το representation capacity όσο και τη μεταφορά.',
      finding: 'Με λιγότερο από ένα εκατομμύριο parameters φτάνει 0.9648 AUC στο primary test set, αλλά μόνο 0.5488 OOD. Άρα η βασική εργασία είναι εύκολη, ενώ η γενίκευση παραμένει σχεδόν στο chance.',
      acc: .9011, auc: .9648, ood: .5488, gap: .3523, robustness: .7901
    },
    {
      key: 'cnn-residual', name: 'cnn-residual', family: 'Residual CNN + SE', subtitle: 'Custom pre-activation residual network με channel attention και EMA weights.',
      hypothesis: 'Μεγαλύτερο βάθος, residual gradient flow και channel attention πρέπει να ξεπεράσουν το απλό CNN.',
      diagram: 'assets/architectures/cnn-residual-diagram.png', evidence: 'assets/evidence/cnn-residual.png', evidenceLabel: 'Grad-CAM του τελευταίου residual stage.',
      science: 'Οι residual συνδέσεις μαθαίνουν διορθώσεις F(x) πάνω σε ένα identity path x, άρα το block υπολογίζει y=x+F(x). Η pre-activation διάταξη BN→ReLU→Conv αφήνει καθαρό gradient highway. Το Squeeze-and-Excitation κάνει channel attention: συνοψίζει κάθε feature map, προβλέπει gating weights και ενισχύει τα channels που μεταφέρουν forensic signal χωρίς spatial pooling.',
      formula: 'y = x + F(BN→ReLU→Conv(x));   s = σ(W₂·ReLU(W₁·GAP(y)));   y′ = s ⊙ y',
      stages: [
        ['RGB 128²', 'Ίδιο input με το baseline ώστε η σύγκριση να απομονώνει την αρχιτεκτονική.'],
        ['Conv stem', 'Full-resolution stem που δεν απορρίπτει νωρίς high-frequency detail.'],
        ['Pre-act blocks', 'Τρία stages από identity residual blocks. Τα stride-2 transitions αυξάνουν channels 64→128→256.'],
        ['SE attention', 'Global channel descriptor → bottleneck MLP → sigmoid gates. Η προσοχή είναι ανά feature channel, όχι ανά pixel.'],
        ['EMA head', 'GAP, dropout, binary head. Η αποθηκευμένη πρόβλεψη χρησιμοποιεί exponential moving average των weights.']
      ],
      specs: { Input: 'RGB 128×128', Normalization: 'Dataset mean/std', Parameters: '≈2.8M', Search: 'Untuned baseline', Loss: 'BCE + smoothing', Explainability: 'Grad-CAM' },
      training: '40 epochs με AdamW, cosine warmup και EMA decay 0.999. Το τελευταίο BN scale κάθε residual branch αρχικοποιείται στο μηδέν, ώστε το δίκτυο να ξεκινά κοντά στην identity function και να μαθαίνει σταδιακά μη γραμμικές διορθώσεις.',
      limitation: 'Η απόδοση δεν είναι έγκυρη απόρριψη του residual design: το pipeline δεν tuned με Optuna και είναι πολύ πιο ευαίσθητο σε learning rate, warmup και EMA decay.',
      finding: 'Παρά τη μεγαλύτερη χωρητικότητα, πέφτει σε 0.8672 AUC και 0.5190 OOD. Η υποαπόδοση έναντι του απλού baseline είναι ένδειξη optimisation failure, όχι ότι τα residual blocks ή το SE βλάπτουν θεωρητικά.',
      acc: .7868, auc: .8672, ood: .5190, gap: .2679, robustness: .7311
    },
    {
      key: 'cnn-finetune', name: 'cnn-finetune', family: 'Transfer learning', subtitle: 'ImageNet-pretrained EfficientNet-B0 με two-stage fine-tuning και discriminative learning rates.',
      hypothesis: 'Ένα ισχυρό ImageNet prior θα μεταφέρει χρήσιμες οπτικές αναπαραστάσεις και θα μειώσει το κόστος εκμάθησης του forensic boundary.',
      diagram: 'assets/architectures/cnn-finetune-diagram.png', evidence: 'assets/evidence/cnn-finetune.png', evidenceLabel: 'Grad-CAM του fine-tuned EfficientNet-B0.',
      science: 'Το EfficientNet-B0 χρησιμοποιεί MBConv inverted bottlenecks, depthwise separable convolutions και Squeeze-and-Excitation για υψηλή parameter efficiency. Η ImageNet προεκπαίδευση παρέχει ήδη φίλτρα για edges, textures και shapes. Το fine-tuning δεν μαθαίνει όραση από την αρχή· επαναπροσανατολίζει ένα ώριμο feature space στο binary forensic task.',
      formula: 'z = EfficientNetImageNet(x);   logit = wᵀ·Dropout(GAP(z));   p(fake)=σ(logit)',
      stages: [
        ['RGB 224²', 'ImageNet normalization ώστε οι activations να παραμένουν στο distribution της προεκπαίδευσης.'],
        ['MBConv backbone', 'Inverted bottleneck expansion → depthwise spatial convolution → projection, με residual όπου επιτρέπεται.'],
        ['SE gates', 'Channel reweighting μέσα στα EfficientNet blocks ενισχύει informative feature maps.'],
        ['Global pooling', 'Το spatial tensor συμπυκνώνεται σε semantic/texture descriptor.'],
        ['EMA binary head', 'Dropout + single logit. Η τελική αποθήκευση χρησιμοποιεί smoothed EMA weights.']
      ],
      specs: { Input: 'RGB 224×224', Normalization: 'ImageNet', Parameters: '≈5M', Search: '12 Optuna trials', Loss: 'Focal γ≈2.94', Explainability: 'Grad-CAM' },
      training: 'Πρώτα παγώνει ο backbone για 3 epochs και εκπαιδεύεται μόνο το νέο head. Έπειτα γίνεται unfreeze με discriminative learning rates: τα πρώτα generic layers αλλάζουν αργά, τα late layers και το head γρηγορότερα. Αυτό μειώνει catastrophic forgetting.',
      limitation: 'Η ισχυρή προσαρμογή στο primary dataset δίνει το μεγαλύτερο generalisation gap του project: εξαιρετικό fit, αλλά έντονη specialization στα training fingerprints.',
      finding: 'Η μεταφορά γνώσης ανεβάζει το ID AUC στο 0.9930, αλλά η OOD accuracy μένει 0.5636. Το αποτέλεσμα δείχνει ότι ένα powerful prior μπορεί να υπερπροσαρμοστεί όταν γίνει full fine-tuning.',
      acc: .9559, auc: .9930, ood: .5636, gap: .3923, robustness: .8591
    },
    {
      key: 'vit-lora', name: 'vit-lora', family: 'Vision Transformer + PEFT', subtitle: 'Frozen ViT-Base με low-rank LoRA updates στα attention projections.',
      hypothesis: 'Global self-attention και περιορισμένη low-rank προσαρμογή θα δώσουν ισχυρό fit με μικρότερο overfitting από full fine-tuning.',
      diagram: 'assets/architectures/vit-lora.png', evidence: 'assets/evidence/vit-lora.png', evidenceLabel: 'Attention rollout από [CLS] προς τα image patches.',
      science: 'Ο Vision Transformer τεμαχίζει την εικόνα σε 16×16 patches, τα μετατρέπει σε tokens και χρησιμοποιεί multi-head self-attention ώστε κάθε patch να αλληλεπιδρά με όλα τα υπόλοιπα. Το [CLS] token συγκεντρώνει global evidence. Η LoRA κρατά παγωμένο κάθε μεγάλο matrix W και μαθαίνει μόνο μια low-rank διόρθωση BA στα q, k, v projections.',
      formula: 'Attention(Q,K,V)=softmax(QKᵀ/√d)V;   W′ = W + BA,   rank(BA)=r≪d',
      stages: [
        ['RGB 224²', 'Η εικόνα χωρίζεται σε 14×14=196 patches των 16×16 pixels.'],
        ['Patch tokens', 'Κάθε patch προβάλλεται σε embedding και προστίθενται positional embeddings και [CLS].'],
        ['Self-attention', 'Τα q/k/v interactions συνδυάζουν τοπική υφή και global συνέπεια σε όλο το frame.'],
        ['LoRA update', 'Μόνο τα B·A matrices λαμβάνουν gradients. Το frozen W διατηρεί το pretrained representation.'],
        ['CLS head', 'Το τελικό [CLS] embedding οδηγεί έναν trainable binary classifier.']
      ],
      specs: { Input: 'RGB 224×224', Normalization: 'ImageNet', Parameters: '≈1.2M trainable / 86M total', Search: '12 Optuna trials', Loss: 'Focal', Explainability: 'Attention rollout' },
      training: 'Μόνο οι LoRA adapters και το classification head εκπαιδεύονται. Το rank r=32 ήταν το μεγαλύτερο διαθέσιμο και κέρδισε στο Optuna search. Στο inference τα adapters συγχωνεύονται στο W, άρα δεν προσθέτουν latency.',
      limitation: 'Η παγκόσμια self-attention είναι ανθεκτική σε θόρυβο, αλλά το global resized view εξακολουθεί να χάνει native-resolution forensic detail που διατηρεί το patch ensemble.',
      finding: 'Είναι το καλύτερο ID μοντέλο (AUC 0.9972), το καλύτερα calibrated (Brier 0.0172) και το πιο robust. Στο OOD πέφτει δεύτερο με 0.6022, επιβεβαιώνοντας ότι PEFT περιορίζει αλλά δεν εξαφανίζει το generator overfit.',
      acc: .9782, auc: .9972, ood: .6022, gap: .3760, robustness: .9228
    },
    {
      key: 'clip-probe', name: 'clip-probe', family: 'Foundation embedding probe', subtitle: 'Frozen CLIP ViT-B/32 encoder, cached 512-D embeddings και trainable MLP head.',
      hypothesis: 'Το frozen semantic manifold του CLIP θα εντοπίσει απόκλιση από τις πραγματικές εικόνες και θα γενικεύσει καλύτερα από low-level detectors.',
      diagram: 'assets/architectures/clip-probe-diagram.png', evidence: 'assets/evidence/clip-probe.png', evidenceLabel: 't-SNE των frozen CLIP embeddings ανά label και generator.',
      science: 'Το CLIP έχει εκπαιδευτεί contrastively ώστε matching image-text pairs να πλησιάζουν σε κοινό embedding space και mismatched pairs να απομακρύνονται. Ο image encoder κωδικοποιεί high-level semantic και distributional structure. Επειδή παραμένει frozen, ο ανιχνευτής μαθαίνει μόνο μια μη γραμμική επιφάνεια απόφασης πάνω στο 512-D manifold.',
      formula: 'z = normalize(CLIPimage(x));   h = MLP(z);   p(fake)=σ(wᵀh)',
      stages: [
        ['RGB 224²', 'CLIP-specific normalization και resize.'],
        ['Frozen ViT-B/32', 'Contrastively pretrained image encoder. Κανένα backbone weight δεν αλλάζει.'],
        ['512-D embedding', 'Ένας σταθερός semantic descriptor αποθηκεύεται μία φορά στο disk cache.'],
        ['MLP probe', 'Linear–ReLU–Dropout blocks μαθαίνουν τη real/fake boundary στο frozen space.'],
        ['Binary logit', 'Το probe δίνει p(fake) χωρίς spatial feature map για Grad-CAM.']
      ],
      specs: { Input: 'RGB 224×224', Normalization: 'CLIP', Parameters: '<1M trainable', Search: '80 Optuna trials', Loss: 'Focal', Explainability: 'Embedding t-SNE' },
      training: 'Η ακριβή κωδικοποίηση γίνεται μία φορά. Όλο το Optuna search και η τελική εκπαίδευση λειτουργούν πάνω σε cached vectors, επιτρέποντας 80 trials. Το backbone δεν μπορεί να overfit μέσω weight updates, αλλά το head μπορεί να εκμεταλλευτεί dataset-specific semantic correlations.',
      limitation: 'Το 512-D vector απορρίπτει spatially local και πολύ υψηλής συχνότητας στοιχεία. Δεν υπάρχει faithful spatial explanation από το probe.',
      finding: 'Η preregistered υπόθεση ότι το CLIP θα γενικεύει καλύτερα δεν επιβεβαιώθηκε. Έφτασε τρίτο στο OOD (0.5837): το semantic signal μεταφέρεται, αλλά λιγότερο από τα native-resolution patch artifacts.',
      acc: .9592, auc: .9930, ood: .5837, gap: .3755, robustness: null
    },
    {
      key: 'two-stream', name: 'two-stream', family: 'Spatial + frequency fusion', subtitle: 'Δύο παράλληλοι CNN κλάδοι για RGB content και luminance log-FFT.',
      hypothesis: 'Το φασματικό view θα προσθέσει συμπληρωματικό evidence στο RGB και η fusion θα ξεπεράσει κάθε stream μόνο του.',
      diagram: 'assets/architectures/two-stream.png', evidence: 'assets/evidence/two-stream.png', evidenceLabel: 'Grad-CAM και component-wise evidence του two-stream model.',
      science: 'Η ίδια εικόνα αναλύεται σε δύο complementary representations. Ο spatial branch διαβάζει περιεχόμενο και υφή στο RGB domain. Ο frequency branch διαβάζει τη log-magnitude του 2-D FFT της luminance, όπου periodic upsampling traces εμφανίζονται πιο άμεσα. Η concatenation fusion μαθαίνει κοινή απόφαση, ενώ auxiliary losses υποχρεώνουν κάθε stream να παραμένει μόνο του predictive.',
      formula: 'F=log(1+|FFT(Y)|);   z=[CNNrgb(x) ‖ CNNfft(F)];   L=Lfused+λ(Lrgb+Lfft)',
      stages: [
        ['RGB 128²', 'Κοινή augmented εικόνα για συγχρονισμένη spatial/frequency παρατήρηση.'],
        ['Spatial CNN', 'Μαθαίνει semantic και texture features απευθείας από RGB.'],
        ['Log-FFT CNN', 'Μετασχηματίζει luminance στο frequency plane και μαθαίνει spectral patterns.'],
        ['Concat fusion', 'Τα δύο feature vectors ενώνονται και ένα fusion head μαθαίνει joint interactions.'],
        ['Auxiliary heads', 'Κάθε branch έχει δικό του logit· το joint loss αποτρέπει branch collapse.']
      ],
      specs: { Input: 'RGB + FFT, 128×128', Normalization: 'Dataset', Parameters: '≈0.78M', Search: '20 Optuna trials', Loss: 'Focal + auxiliary BCE', Explainability: 'Grad-CAM + branch scores' },
      training: 'Το FFT υπολογίζεται on-the-fly μετά το augmentation, ώστε οι δύο κλάδοι να βλέπουν ακριβώς την ίδια εικόνα. Το λ≈0.21 του auxiliary loss ισορροπεί independent branch skill και fused performance.',
      limitation: 'Ο frequency branch είναι ασθενέστερος μόνος του και εξαιρετικά ευαίσθητος σε additive noise. Η concat fusion δεν μπορεί να αγνοεί δυναμικά ένα αναξιόπιστο stream ανά εικόνα.',
      finding: 'Η fusion υπερτερεί από κάθε branch μεμονωμένα, αλλά το OOD score παραμένει 0.5264. Το frequency cue είναι complementary, όχι επαρκές από μόνο του.',
      acc: .8977, auc: .9609, ood: .5264, gap: .3713, robustness: .7869
    },
    {
      key: 'freqcross', name: 'freqcross', family: 'Three-branch attention fusion', subtitle: 'Spatial, 2-D frequency και radial-spectrum branches με learned softmax gate.',
      hypothesis: 'Ένα adaptive gate πάνω σε RGB, 2-D FFT και radial profile θα επιλέγει το πιο αξιόπιστο cue ανά εικόνα.',
      diagram: 'assets/architectures/freqcross.png', evidence: 'assets/evidence/freqcross.png', evidenceLabel: 'Per-image fusion attention και branch contribution.',
      science: 'Το freqcross επεκτείνει το two-stream με τρίτη άποψη: το azimuthally averaged radial power profile. Η differentiable radial layer ομαδοποιεί Fourier energy σε bins ακτίνας και ένα MLP διαβάζει την καμπύλη fall-off. Αντί για σταθερή concatenation, ένα attention gate προβλέπει softmax weights ανά εικόνα και αναμειγνύει τα branch representations δυναμικά.',
      formula: 'z=Σᵢ αᵢfᵢ,   α=softmax(g(fspatial, fFFT, fradial)),   Σᵢαᵢ=1',
      stages: [
        ['RGB 128²', 'Μία εικόνα τροφοδοτεί τρεις συγχρονισμένες views.'],
        ['Spatial branch', 'RGB CNN για content και local texture.'],
        ['Frequency branch', '2-D log-FFT CNN για grid-aligned spectral structure.'],
        ['Radial branch', 'Differentiable radial binning + MLP για rotation-agnostic spectral fall-off.'],
        ['Attention fusion', 'Softmax gate κατανέμει βάρος ανά branch και εικόνα πριν από το final logit.']
      ],
      specs: { Input: 'RGB + FFT + radial, 128²', Normalization: 'Dataset', Parameters: '≈0.82M', Search: '20 Optuna trials', Loss: 'BCE + auxiliary heads', Explainability: 'Fusion weights + Grad-CAM' },
      training: 'Τα τρία branches και το gate εκπαιδεύονται end-to-end. Auxiliary heads διατηρούν κάθε branch predictive. Το Optuna επέλεξε coarse 48-bin radial profile, ένδειξη ότι το χρήσιμο signal βρίσκεται στη συνολική spectral fall-off και όχι σε πολύ λεπτή binning.',
      limitation: 'Το attention gate αυξάνει την ερμηνευσιμότητα, αλλά δεν διορθώνει το distribution shift όταν αλλάζει το ίδιο το spectral fingerprint.',
      finding: 'Είναι το καλύτερο 128-px model σε ID AUC (0.9651), με μέση attention περίπου 0.61 spatial, 0.29 radial και 0.10 FFT. Στο OOD φτάνει 0.5526: μικρή αλλά πραγματική μεταφορά.',
      acc: .9003, auc: .9651, ood: .5526, gap: .3476, robustness: .7860
    },
    {
      key: 'srm-noise', name: 'srm-noise', family: 'Forensic residual detector', subtitle: 'Fixed SRM high-pass bank και learnable Bayar constrained convolution.',
      hypothesis: 'Αν αφαιρεθεί το semantic content, το noise residual θα αποκαλύψει ένα generator fingerprint που μεταφέρεται καλύτερα.',
      diagram: 'assets/architectures/srm-noise-diagram2.png', evidence: 'assets/evidence/srm-noise.png', evidenceLabel: 'RGB, SRM και Bayar residual maps που αποτελούν την πραγματική είσοδο του classifier.',
      science: 'Αντί να ταξινομεί το ορατό περιεχόμενο, το pipeline αφαιρεί το προβλέψιμο low-frequency μέρος και αναλύει το prediction error. Τα fixed SRM kernels παρέχουν ισχυρό forensic prior. Η Bayar convolution επιβάλλει κεντρικό coefficient −1 και άθροισμα γειτονικών coefficients +1, ώστε να παραμένει residual operator ενώ προσαρμόζεται στα δεδομένα.',
      formula: 'r(p)=Σq≠p wq·x(q) − x(p),   wp=−1,   Σq≠p wq=+1',
      stages: [
        ['RGB 128²', 'Το content δεν ταξινομείται απευθείας.'],
        ['SRM bank', 'Τρία fixed high-pass filters απομονώνουν γνωστά steganalysis residuals.'],
        ['Bayar conv', 'Learnable constrained filters προσαρμόζουν το residual χωρίς να γίνουν ordinary convolution.'],
        ['Residual concat', 'Fixed και learned noise maps ενώνονται σε multi-channel forensic tensor.'],
        ['Feature CNN', 'Ένα μικρό CNN ταξινομεί τη στατιστική δομή του residual και παράγει p(fake).']
      ],
      specs: { Input: 'Residual maps, 128×128', Normalization: 'Dataset', Parameters: '≈0.39M', Search: '24 Optuna trials', Loss: 'BCE', Explainability: 'Native residual maps' },
      training: 'Ο SRM buffer δεν εκπαιδεύεται. Τα Bayar filters επανακανονικοποιούνται σε κάθε forward pass ώστε οι constraints να ισχύουν πάντα. Μόνο ο Bayar branch και το downstream CNN λαμβάνουν gradients.',
      limitation: 'Η αφαίρεση του semantic content πετά και χρήσιμο signal. Τα residuals είναι ιδιαίτερα εύθραυστα σε Gaussian noise, το οποίο καλύπτει ακριβώς το forensic pattern.',
      finding: 'Με μόλις 0.39M parameters πετυχαίνει 0.9518 AUC, επιβεβαιώνοντας ότι generator signal υπάρχει στο noise residual. Η OOD accuracy 0.5231 δείχνει όμως ότι το residual παραμένει generator-dependent.',
      acc: .8824, auc: .9518, ood: .5231, gap: .3593, robustness: .7601
    },
    {
      key: 'patch-ensemble', name: 'patch-ensemble', family: 'Multiple-instance learning', subtitle: 'Native-resolution patch bags, shared EfficientNet encoder και gated-attention MIL.',
      hypothesis: 'Το transferable evidence βρίσκεται σε native-resolution local texture και δεν πρέπει να αραιώνεται από global resize ή pooling.',
      diagram: 'assets/architectures/patch-ensemble-diagram.png', evidence: 'assets/evidence/patch-ensemble.png', evidenceLabel: 'Faithful per-patch MIL attention weights.',
      science: 'Στο multiple-instance learning κάθε εικόνα είναι bag από K patches και το label υπάρχει μόνο σε επίπεδο bag. Ένας shared encoder παράγει feature fk για κάθε patch. Η gated attention συνδυάζει tanh και sigmoid projections, προβλέπει κανονικοποιημένο βάρος ak και σχηματίζει weighted sum. Έτσι ένα τοπικό artifact μπορεί να κυριαρχήσει χωρίς να αραιωθεί από bland regions.',
      formula: 'aₖ=softmax(wᵀ[tanh(Vfₖ)⊙σ(Ufₖ)]);   z=Σₖaₖfₖ',
      stages: [
        ['Native image', 'Το pipeline παρακάμπτει την global 256² cache και διατηρεί τα original pixels.'],
        ['Patch bag', 'K=6 random native crops στο train. Ο current eval loader επιστρέφει 4 deterministic corner crops, παρότι το tuned K είναι 6.'],
        ['Shared EffNet-B0', 'Το ίδιο pretrained encoder εφαρμόζεται batched σε όλα τα patches.'],
        ['Gated MIL', 'Input-dependent attention επιλέγει τα patches με το ισχυρότερο forensic evidence.'],
        ['Bag prediction', 'Το attention-weighted image vector οδηγεί το single-logit head.']
      ],
      specs: { Input: '6 train / 4 eval native 224² crops', Normalization: 'ImageNet', Parameters: '≈4.7M', Search: '8 Optuna trials', Loss: 'Focal γ≈1.40', Explainability: 'Faithful MIL weights' },
      training: 'Κάθε image step απαιτεί K backbone passes, γι’ αυτό το search περιορίστηκε σε 8 ακριβά trials. Το Optuna επέλεξε το μέγιστο K=6 και MIL hidden=256, υποδεικνύοντας ότι περισσότερα local views και richer gating ήταν χρήσιμα.',
      limitation: 'Παρακάμπτει το 256² shortcut-neutralised cache, άρα native resolution και class-correlated source resolution παραμένουν πιθανό confound. Επιπλέον, το current eval bag έχει 4 corner crops αντί του tuned K=6. Δεν εντάχθηκε στο robustness sweep.',
      finding: 'Είναι το καλύτερο μετρημένο OOD pipeline: 0.6775 accuracy και ουσιαστικό gap 0.2906. Το residual έχει αριθμητικά μικρότερο gap 0.2679 μόνο λόγω χαμηλού ID ceiling. Το αποτέλεσμα στηρίζει, αλλά δεν αποδεικνύει μόνο του, την υπόθεση του native local texture· απαιτεί matched-resolution ablation.',
      acc: .9681, auc: .9963, ood: .6775, gap: .2906, robustness: null
    },
    {
      key: 'dire-recon', name: 'dire-recon', family: 'Reconstruction-based detection', subtitle: 'DDIM inversion/reconstruction με Stable Diffusion v1.5 και CNN πάνω στο cached error map.',
      hypothesis: 'Μια diffusion-generated εικόνα θα ανακατασκευάζεται πιο πιστά από το SD v1.5 manifold από ό,τι μια πραγματική φωτογραφία.',
      diagram: 'assets/architectures/dire-recon-diagram2.png', evidence: 'assets/evidence/dire-recon.png', evidenceLabel: 'DIRE reconstruction-error maps για real και generated images.',
      science: 'Η DIRE υπόθεση είναι γεωμετρική: μια diffusion-generated εικόνα βρίσκεται κοντά στο manifold του generative model που την ανακατασκευάζει, ενώ μια πραγματική φωτογραφία βρίσκεται περισσότερο off-manifold. Με DDIM inversion βρίσκουμε latent noise, εκτελούμε reverse diffusion και μετράμε το per-pixel |x−x̂|. Έπειτα ένα CNN μαθαίνει τη δομή αυτού του error map.',
      formula: 'zT=DDIMinvert(x);   x̂=SD1.5decode(reverse(zT));   DIRE(x)=|x−x̂|',
      stages: [
        ['Input image', 'Η αρχική RGB εικόνα χρησιμοποιείται μόνο για reconstruction.'],
        ['DDIM inversion', '20 deterministic diffusion steps χαρτογραφούν την εικόνα σε latent noise.'],
        ['Reconstruction', 'Το Stable Diffusion v1.5 reverse process παράγει x̂ πάνω στο δικό του manifold.'],
        ['DIRE map', 'Το absolute pixel error |x−x̂| μετατρέπεται σε cached 256²×3 forensic image.'],
        ['EffNet classifier', 'Ένα ordinary EfficientNet-B0 ταξινομεί το error map, όχι το original RGB.']
      ],
      specs: { Input: 'DIRE map → 224×224', Normalization: 'ImageNet', Parameters: '≈5M classifier', Search: '20 Optuna trials', Loss: 'BCE', Explainability: 'DIRE map is the input' },
      training: 'Το ακριβό diffusion pass εκτελείται μία φορά ανά εικόνα και γίνεται cache. Μετά, tuning και training είναι ένα συνηθισμένο EfficientNet fine-tune πάνω σε error maps. Η σχεδίαση front-loads το compute, αλλά απαιτεί Stable Diffusion για κάθε νέο input.',
      limitation: 'Τα ID metrics βασίζονται σε subsample 2,000 εικόνων και δεν είναι head-to-head συγκρίσιμα. Το error signal είναι δεμένο στο manifold του SD v1.5.',
      finding: 'Το AUC 0.9399 επιβεβαιώνει ότι reconstruction error είναι learnable signal. Η OOD accuracy 0.5415 είναι χαμηλή και μεταφέρεται καλύτερα σε SD-adjacent generators, χειρότερα σε GAN-era models.',
      acc: .8730, auc: .9399, ood: .5415, gap: .3315, robustness: null
    }
  ];

  const protocolData = {
    id: {
      title: 'P1 · Επίδοση εντός κατανομής',
      copy: 'Train και test προέρχονται από τις ίδιες generator families. Το protocol αποδεικνύει ότι το μοντέλο μπορεί να λύσει τη βασική εργασία, αλλά δεν διαχωρίζει transferable detection από fingerprint memorisation.',
      facts: [['Test set', '11,963'], ['Primary metric', 'AUC-ROC'], ['Calibration', 'Brier score'], ['Threshold', 'Validation-tuned']],
      image: 'assets/indist-bars.png', caption: 'Η ισχυρή τετράδα 224-px ξεπερνά 0.99 AUC.'
    },
    ood: {
      title: 'P2 · Γενίκευση μεταξύ generators',
      copy: 'Κάθε pipeline αξιολογείται untouched στα επτά generator subsets του tiny-genimage, χωρίς καμία χρήση τους σε training, tuning ή threshold selection. Έξι generator names είναι νέα· το Midjourney επανεμφανίζεται ως ανεξάρτητη OOD συλλογή, άρα το protocol μετρά ταυτόχρονα generator και dataset/content shift.',
      facts: [['OOD images', '34,998'], ['Generator subsets', '7 held out'], ['Random baseline', '0.500'], ['Maximum OOD', '0.6775']],
      image: 'assets/ood-random.png', caption: 'Το καλύτερο μοντέλο βρίσκεται μόλις +0.177 πάνω από chance.'
    },
    robustness: {
      title: 'P3 · Ανθεκτικότητα σε αλλοιώσεις',
      copy: 'Τέσσερις αλλοιώσεις εφαρμόζονται σε πέντε επίπεδα, σε subsample των επτά image-family pipelines. Επειδή αποκλείστηκαν από τα training augmentations, οι καμπύλες μετρούν out-of-the-box robustness και όχι rehearsed invariance.',
      facts: [['JPEG Q60 drop', '0.003'], ['Blur σ=2 drop', '≈0'], ['Downsample drop', '0.221'], ['Noise drop', '0.350']],
      image: 'assets/robustness-curves.png', caption: 'Noise και downsampling, όχι JPEG/blur, είναι οι ουσιαστικές απειλές.'
    },
    xai: {
      title: 'XAI · Εξηγήσεις προσαρμοσμένες στην αρχιτεκτονική',
      copy: 'Δεν επιβάλλεται μία μέθοδος explanation σε όλα τα μοντέλα. Χρησιμοποιείται ο μηχανισμός που αντιστοιχεί στην εσωτερική αναπαράσταση κάθε pipeline. Οι εικόνες είναι διαγνωστικά shortcut audits, όχι αιτιώδεις εγγυήσεις πιστότητας.',
      facts: [['CNN', 'Grad-CAM'], ['ViT', 'Attention rollout'], ['Patch MIL', 'Attention weights'], ['Forensics', 'Residual / DIRE maps']],
      image: 'assets/gradcam-gallery.png', caption: 'Οι εξηγήσεις ελέγχουν borders, backgrounds και collection shortcuts.'
    }
  };

  const resultData = {
    id: {
      title: 'Υψηλή διαχωριστική ικανότητα εντός κατανομής', insight: 'Μέγιστο ID AUC: 0.9972',
      copy: 'ViT-LoRA, patch-ensemble, cnn-finetune και clip-probe ξεπερνούν το 0.99 AUC. Η κατάταξη εξηγείται κυρίως από input resolution × prior strength. Το DIRE είναι ένδειξη μόνο: αξιολογείται σε subsample 2,000 αντί για το κοινό n=11,963.',
      image: 'assets/indist-bars.png', caption: 'ID test n=11,963 για 9 pipelines· DIRE n=2,000 και δεν είναι άμεσα συγκρίσιμο.'
    },
    ood: {
      title: 'Μεταβολή της κατάταξης υπό OOD αξιολόγηση', insight: 'Μέγιστη OOD accuracy: 0.6775',
      copy: 'Το patch ensemble υπερέχει του ViT-LoRA κατά 7.5 percentage points. Τα μοντέλα χαμηλότερης κατάταξης βρίσκονται λίγες μονάδες πάνω από το random baseline. Το 0.2906 είναι το μικρότερο ουσιαστικό gap· το residual έχει 0.2679 λόγω χαμηλού ID ceiling, όχι λόγω ισχυρής μεταφοράς.',
      image: 'assets/ood-random.png', caption: 'OOD accuracy και lift over random.'
    },
    generators: {
      title: 'Η δυσκολία εξαρτάται από τον generator', insight: 'Δυσκολότερα subsets: VQDM / BigGAN',
      copy: 'Οι στήλες της heatmap φωτίζονται και σκοτεινιάζουν μαζί. Η distributional distance κυριαρχεί έναντι ενός model-specific blind spot. Το Midjourney είναι ευκολότερο ακριβώς επειδή υπάρχει και στο primary generator mix, άρα δεν αποτελεί καθαρά unseen family.',
      image: 'assets/ood-heatmap.png', caption: 'Per-generator OOD accuracy για τα δέκα pipelines.'
    },
    robustness: {
      title: 'Η ανθεκτικότητα εξαρτάται από τον τύπο αλλοίωσης', insight: 'Μέση πτώση με θόρυβο: 0.350',
      copy: 'Τα frequency και residual pipelines χάνουν το signal αμέσως με additive noise. Το ViT-LoRA, που διαβάζει global structure, είναι το μόνο με worst-case accuracy πάνω από 0.70.',
      image: 'assets/robustness-curves.png', caption: 'Accuracy ως συνάρτηση perturbation strength.'
    }
  };

  const state = {
    view: 'overview',
    rq: 0,
    dataset: 'primary',
    dataTopic: 'samples',
    architecture: 'cnn-scratch',
    archTab: 'mechanism',
    stage: 0,
    protocol: 'id',
    result: 'id'
  };

  const byKey = (key) => architectures.find((item) => item.key === key) || architectures[0];
  const fixed = (value) => value.toFixed(4);
  const sectionIndex = () => sections.findIndex((item) => item.key === state.view);

  function setView(key, updateHash = true) {
    if (!sections.some((item) => item.key === key)) key = 'overview';
    state.view = key;
    document.querySelectorAll('.view').forEach((view) => {
      const active = view.dataset.view === key;
      view.classList.toggle('active', active);
      view.setAttribute('aria-hidden', String(!active));
      if (active) view.querySelector('.view-scroll, .architecture-detail')?.scrollTo?.({ top: 0 });
    });
    document.querySelectorAll('.nav-tab[data-view-target]').forEach((button) => {
      const active = button.dataset.viewTarget === key;
      button.classList.toggle('active', active);
      button.setAttribute('aria-current', active ? 'page' : 'false');
    });
    document.querySelector(`.nav-tab[data-view-target="${key}"]`)?.scrollIntoView({ block: 'nearest', inline: 'center' });
    const index = sectionIndex();
    document.getElementById('section-index').textContent = String(index + 1).padStart(2, '0');
    document.getElementById('section-name').textContent = sections[index].label;
    document.getElementById('previous-section').disabled = index === 0;
    document.getElementById('next-section').disabled = index === sections.length - 1;
    document.title = `${sections[index].label} | Deepfake Detection`;
    if (updateHash) writeHash();
  }

  function writeHash() {
    const hash = state.view === 'architectures' ? `#architectures/${state.architecture}` : `#${state.view}`;
    if (location.hash !== hash) history.replaceState(null, '', hash);
  }

  function readHash() {
    const raw = location.hash.replace(/^#/, '');
    const [view, model] = raw.split('/');
    if (view === 'architectures' && architectures.some((item) => item.key === model)) state.architecture = model;
    setView(sections.some((item) => item.key === view) ? view : 'overview', false);
    if (state.view === 'architectures') renderArchitecture(false);
  }

  function renderResearchQuestion() {
    const item = researchQuestions[state.rq];
    document.querySelectorAll('[data-rq]').forEach((button) => {
      const active = Number(button.dataset.rq) === state.rq;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
    });
    document.getElementById('rq-detail').innerHTML = `<h3>${item.title}</h3><p><strong>Ερώτηση:</strong> ${item.question}</p><p><strong>Απάντηση:</strong> ${item.conclusion}</p>`;
  }

  function renderOverviewModels() {
    const container = document.getElementById('overview-model-map');
    container.innerHTML = architectures.map((model) => `<button type="button" data-model-jump="${model.key}"><strong>${model.name}</strong><span>${model.family}</span></button>`).join('');
  }

  function renderDataLab() {
    const dataset = datasetData[state.dataset];
    const topic = dataTopics[state.dataTopic];
    document.querySelectorAll('[data-dataset]').forEach((button) => button.classList.toggle('active', button.dataset.dataset === state.dataset));
    document.querySelectorAll('[data-dataset]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.dataset === state.dataset)));
    document.querySelectorAll('[data-data-topic]').forEach((button) => {
      const active = button.dataset.dataTopic === state.dataTopic;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
    });
    document.getElementById('dataset-facts').innerHTML = `<h2>${dataset.title}</h2><p>${dataset.role}</p><dl><dt>Μέγεθος</dt><dd>${dataset.size}</dd><dt>Ισορροπία</dt><dd>${dataset.balance}</dd><dt>Generators</dt><dd>${dataset.generators}</dd><dt>Real source</dt><dd>${dataset.realSource}</dd></dl>`;
    const image = document.getElementById('data-figure');
    if (state.dataTopic === 'samples') image.src = dataset.samples;
    if (state.dataTopic === 'frequency') image.src = dataset.spectrum;
    if (state.dataTopic === 'embedding') image.src = dataset.embedding;
    if (state.dataTopic === 'shortcut') image.src = dataset.properties;
    image.alt = `${topic.title} — ${dataset.title}`;
    document.getElementById('data-caption').textContent = `${topic.caption} Dataset: ${dataset.title}.`;
    document.getElementById('data-analysis').innerHTML = `<h2>${topic.title}</h2><p>${topic.analysis}</p><ul>${topic.bullets.map((text) => `<li>${text}</li>`).join('')}</ul>`;
    prepareZoomable(image);
  }

  function renderArchitectureList() {
    const list = document.getElementById('architecture-list');
    list.innerHTML = architectures.map((model) => `<button type="button" role="option" data-architecture="${model.key}" aria-selected="${model.key === state.architecture}"><strong>${model.name}</strong><span>${model.family}</span></button>`).join('');
  }

  function renderArchitecture(updateHash = true) {
    const model = byKey(state.architecture);
    document.querySelectorAll('[data-architecture]').forEach((button) => {
      const active = button.dataset.architecture === model.key;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
    });
    document.querySelector('.architecture-detail')?.scrollTo?.({ top: 0 });
    document.getElementById('arch-family').textContent = model.family;
    document.getElementById('arch-name').textContent = model.name;
    document.getElementById('arch-subtitle').textContent = model.subtitle;
    document.getElementById('arch-hypothesis').textContent = model.hypothesis;
    document.getElementById('arch-auc').textContent = fixed(model.auc);
    document.getElementById('arch-ood').textContent = fixed(model.ood);
    document.getElementById('arch-gap').textContent = fixed(model.gap);
    const diagram = document.getElementById('arch-diagram');
    diagram.src = model.diagram;
    diagram.alt = `Διάγραμμα αρχιτεκτονικής ${model.name}`;
    document.getElementById('arch-diagram-caption').textContent = `Signal flow του ${model.name}, όπως τεκμηριώνεται στο report.`;
    document.getElementById('arch-science').textContent = model.science;
    document.getElementById('arch-formula').textContent = model.formula;
    document.getElementById('arch-specs').innerHTML = Object.entries(model.specs).map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join('');
    document.getElementById('arch-training').textContent = model.training;
    document.getElementById('arch-limitation').innerHTML = `<strong>Επιστημονικός περιορισμός:</strong> ${model.limitation}`;
    const evidence = document.getElementById('arch-evidence-image');
    evidence.src = model.evidence;
    evidence.alt = `${model.evidenceLabel} ${model.name}`;
    document.getElementById('arch-evidence-caption').textContent = model.evidenceLabel;
    document.getElementById('arch-finding').textContent = model.finding;
    document.getElementById('arch-results-list').innerHTML = `<dt>ID accuracy</dt><dd>${fixed(model.acc)}</dd><dt>ID AUC</dt><dd>${fixed(model.auc)}</dd><dt>OOD accuracy</dt><dd>${fixed(model.ood)}</dd><dt>Generalisation gap</dt><dd>${fixed(model.gap)}</dd><dt>Mean perturbed accuracy</dt><dd>${model.robustness === null ? 'n/a' : fixed(model.robustness)}</dd>`;
    state.stage = 0;
    renderStages();
    setArchitectureTab(state.archTab);
    prepareZoomable(diagram);
    prepareZoomable(evidence);
    if (updateHash) writeHash();
  }

  function renderStages() {
    const model = byKey(state.architecture);
    document.getElementById('stage-buttons').innerHTML = model.stages.map((stage, index) => `<button type="button" role="tab" data-stage="${index}" class="${index === state.stage ? 'active' : ''}" aria-selected="${index === state.stage}">${index + 1}. ${stage[0]}</button>`).join('');
    const stage = model.stages[state.stage];
    document.getElementById('stage-detail').innerHTML = `<strong>${stage[0]}</strong><p>${stage[1]}</p>`;
  }

  function setArchitectureTab(tab) {
    state.archTab = ['mechanism', 'training', 'evidence'].includes(tab) ? tab : 'mechanism';
    document.querySelectorAll('[data-arch-tab]').forEach((button) => {
      const active = button.dataset.archTab === state.archTab;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
    });
    document.querySelectorAll('[data-arch-panel]').forEach((panel) => panel.classList.toggle('active', panel.dataset.archPanel === state.archTab));
  }

  function populateCompare() {
    const options = architectures.map((model) => `<option value="${model.key}">${model.name}</option>`).join('');
    const a = document.getElementById('compare-a');
    const b = document.getElementById('compare-b');
    a.innerHTML = options;
    b.innerHTML = options;
    a.value = state.architecture;
    b.value = state.architecture === 'patch-ensemble' ? 'vit-lora' : 'patch-ensemble';
    renderCompare();
  }

  function renderCompare() {
    const a = byKey(document.getElementById('compare-a').value);
    const b = byKey(document.getElementById('compare-b').value);
    const winner = a.ood === b.ood ? null : (a.ood > b.ood ? a : b);
    const idWinner = a.auc === b.auc ? null : (a.auc > b.auc ? a : b);
    let conclusion = 'Επιλέξτε δύο διαφορετικά pipelines για συγκριτική ερμηνεία.';
    if (a.key !== b.key && idWinner?.key === winner?.key) conclusion = `${winner.name} υπερέχει τόσο σε ID AUC όσο και σε OOD accuracy. Η σύγκριση πρέπει να διαβαστεί μαζί με το input representation και τα experimental caveats.`;
    if (a.key !== b.key && idWinner?.key !== winner?.key) conclusion = `${idWinner ? `${idWinner.name} έχει υψηλότερο ID AUC` : 'Τα ID AUC είναι ισοδύναμα'}, ενώ ${winner ? `${winner.name} έχει υψηλότερη OOD accuracy` : 'οι OOD accuracies είναι ισοδύναμες'}. Η σύγκριση πρέπει να διαβαστεί μαζί με το input representation και τα experimental caveats.`;
    document.getElementById('compare-content').innerHTML = `<table class="compare-table"><thead><tr><th>Χαρακτηριστικό</th><th>${a.name}</th><th>${b.name}</th></tr></thead><tbody><tr><td>Οικογένεια</td><td>${a.family}</td><td>${b.family}</td></tr><tr><td>Input</td><td>${a.specs.Input}</td><td>${b.specs.Input}</td></tr><tr><td>Trainable parameters</td><td>${a.specs.Parameters}</td><td>${b.specs.Parameters}</td></tr><tr><td>ID AUC</td><td>${fixed(a.auc)}</td><td>${fixed(b.auc)}</td></tr><tr><td>OOD accuracy</td><td>${fixed(a.ood)}</td><td>${fixed(b.ood)}</td></tr><tr><td>Generalisation gap</td><td>${fixed(a.gap)}</td><td>${fixed(b.gap)}</td></tr><tr><td>Explainability</td><td>${a.specs.Explainability}</td><td>${b.specs.Explainability}</td></tr></tbody></table><p class="compare-conclusion">${conclusion}</p>`;
  }

  function renderProtocol() {
    const item = protocolData[state.protocol];
    document.querySelectorAll('[data-protocol]').forEach((button) => {
      const active = button.dataset.protocol === state.protocol;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
    });
    document.getElementById('protocol-copy').innerHTML = `<h2>${item.title}</h2><p>${item.copy}</p><dl>${item.facts.map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`).join('')}</dl>`;
    const image = document.getElementById('protocol-image');
    image.src = item.image;
    image.alt = item.title;
    document.getElementById('protocol-caption').textContent = item.caption;
    prepareZoomable(image);
  }

  function renderResults() {
    const item = resultData[state.result];
    document.querySelectorAll('[data-result]').forEach((button) => {
      const active = button.dataset.result === state.result;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
    });
    const image = document.getElementById('results-image');
    image.src = item.image;
    image.alt = item.title;
    document.getElementById('results-caption').textContent = item.caption;
    document.getElementById('results-analysis').innerHTML = `<h2>${item.title}</h2><strong class="big-insight">${item.insight}</strong><p>${item.copy}</p>`;
    prepareZoomable(image);
  }

  function prepareZoomable(image) {
    image.tabIndex = 0;
    image.setAttribute('role', 'button');
    image.setAttribute('aria-label', `${image.alt}. Μεγέθυνση εικόνας.`);
    image.setAttribute('aria-haspopup', 'dialog');
    image.setAttribute('aria-controls', 'image-dialog');
  }

  function openImage(image) {
    const dialog = document.getElementById('image-dialog');
    const target = document.getElementById('dialog-image');
    target.src = image.currentSrc || image.src;
    target.alt = image.alt;
    dialog.showModal();
  }

  function moveSection(delta) {
    const next = Math.min(Math.max(sectionIndex() + delta, 0), sections.length - 1);
    setView(sections[next].key);
  }

  document.addEventListener('click', (event) => {
    const viewButton = event.target.closest('[data-view-target]');
    if (viewButton) setView(viewButton.dataset.viewTarget);

    const modelJump = event.target.closest('[data-model-jump]');
    if (modelJump) {
      state.architecture = modelJump.dataset.modelJump;
      setView('architectures');
      renderArchitectureList();
      renderArchitecture();
    }

    const architectureButton = event.target.closest('[data-architecture]');
    if (architectureButton) {
      state.architecture = architectureButton.dataset.architecture;
      state.archTab = 'mechanism';
      renderArchitecture();
    }

    const stageButton = event.target.closest('[data-stage]');
    if (stageButton) {
      state.stage = Number(stageButton.dataset.stage);
      renderStages();
    }

    const zoomable = event.target.closest('img.zoomable');
    if (zoomable) openImage(zoomable);
  });

  document.addEventListener('keydown', (event) => {
    const tag = event.target.tagName;
    if (tag === 'SELECT' || tag === 'INPUT' || tag === 'TEXTAREA' || document.querySelector('dialog[open]')) return;
    if (event.target.matches('img.zoomable') && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      openImage(event.target);
      return;
    }
    if (event.key === 'ArrowRight' && !event.target.closest('[role="tablist"]')) moveSection(1);
    if (event.key === 'ArrowLeft' && !event.target.closest('[role="tablist"]')) moveSection(-1);
    if (event.key.toLowerCase() === 'f') document.getElementById('fullscreen').click();
  });

  document.querySelectorAll('[data-rq]').forEach((button) => button.addEventListener('click', () => { state.rq = Number(button.dataset.rq); renderResearchQuestion(); }));
  document.querySelectorAll('[data-dataset]').forEach((button) => button.addEventListener('click', () => { state.dataset = button.dataset.dataset; renderDataLab(); }));
  document.querySelectorAll('[data-data-topic]').forEach((button) => button.addEventListener('click', () => { state.dataTopic = button.dataset.dataTopic; renderDataLab(); }));
  document.querySelectorAll('[data-arch-tab]').forEach((button) => button.addEventListener('click', () => setArchitectureTab(button.dataset.archTab)));
  document.querySelectorAll('[data-protocol]').forEach((button) => button.addEventListener('click', () => { state.protocol = button.dataset.protocol; renderProtocol(); }));
  document.querySelectorAll('[data-result]').forEach((button) => button.addEventListener('click', () => { state.result = button.dataset.result; renderResults(); }));

  document.getElementById('previous-section').addEventListener('click', () => moveSection(-1));
  document.getElementById('next-section').addEventListener('click', () => moveSection(1));
  document.getElementById('fullscreen').addEventListener('click', () => document.fullscreenElement ? document.exitFullscreen?.() : document.documentElement.requestFullscreen?.());
  document.addEventListener('fullscreenchange', () => {
    const button = document.getElementById('fullscreen');
    button.title = document.fullscreenElement ? 'Έξοδος από πλήρη οθόνη' : 'Πλήρης οθόνη';
    button.setAttribute('aria-label', button.title);
  });

  document.getElementById('compare-open').addEventListener('click', () => {
    populateCompare();
    document.getElementById('compare-open').setAttribute('aria-expanded', 'true');
    document.getElementById('compare-dialog').showModal();
  });
  document.getElementById('compare-close').addEventListener('click', () => document.getElementById('compare-dialog').close());
  document.getElementById('compare-a').addEventListener('change', renderCompare);
  document.getElementById('compare-b').addEventListener('change', renderCompare);
  document.getElementById('image-close').addEventListener('click', () => document.getElementById('image-dialog').close());
  document.getElementById('compare-dialog').addEventListener('close', () => document.getElementById('compare-open').setAttribute('aria-expanded', 'false'));
  window.addEventListener('hashchange', readHash);

  document.querySelectorAll('img.zoomable').forEach(prepareZoomable);
  renderOverviewModels();
  renderResearchQuestion();
  renderDataLab();
  renderArchitectureList();
  renderArchitecture(false);
  renderProtocol();
  renderResults();
  readHash();
})();
