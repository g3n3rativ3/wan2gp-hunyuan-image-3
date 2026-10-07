# HunyuanImage 3.0 Instruct-Distil pour WanGP

Plugin de modèle pour [WanGP](https://github.com/deepbeepmeep/Wan2GP) qui ajoute **HunyuanImage-3.0 Instruct-Distil** de Tencent.

Il s'agit d'un *mixture-of-experts* de 80 milliards de paramètres, dont 13 milliards actifs par token, utilisé ici en poids **W4A8**. Le modèle sait faire trois choses :

- **texte vers image** en 8 étapes ;
- **édition d'image** (« transforme cette photo en aquarelle ») ;
- **fusion multi-images**, avec jusqu'à 3 images de référence.

Ce plugin est un portage du pack ComfyUI [PedroMarinhoDev/ComfyUI-HunyuanImage3](https://github.com/PedroMarinhoDev/ComfyUI-HunyuanImage3). Seule la version **Instruct-Distil (8 étapes)** est portée. La réécriture de prompt (*prompt rewriting*) n'est pas incluse.

## Matériel nécessaire

| | |
|---|---|
| GPU | NVIDIA, 12 Go de VRAM ou plus |
| RAM système | environ **50 Go libres** : les poids sont gardés en RAM puis envoyés au GPU couche par couche à chaque étape |
| Disque | environ 46 Go (44 Go de transformer, plus le VAE et la tour de vision) |

Le modèle est limité par le transfert des poids, pas par le calcul. La quantité de RAM et la vitesse du bus PCIe comptent plus que la taille du GPU.

## Installation

1. Décompressez l'archive dans le dossier `plugins/` de WanGP. Vous devez obtenir `plugins/wan2gp-hunyuan-image-3/plugin_info.json`.
2. Lancez WanGP, ouvrez l'onglet **Plugins**, cochez **HunyuanImage 3.0 Instruct-Distil**, puis cliquez sur *Save Settings*.
3. **Redémarrez WanGP.** Le modèle apparaît dans la famille **HunyuanImage 3.0**.

Rien n'est à installer avec pip : le plugin n'utilise que des paquets déjà présents dans WanGP.

## Téléchargement automatique des modèles

Au **premier rendu**, WanGP télécharge les fichiers depuis [PedroMarinhoDev/HunyuanImage-3.0-Instruct-Distil-ComfyUI](https://huggingface.co/PedroMarinhoDev/HunyuanImage-3.0-Instruct-Distil-ComfyUI). Ils sont placés dans le dossier de modèles choisi dans **Configuration** :

| Fichier | Emplacement |
|---|---|
| `hunyuan_image_3_instruct_distil_w4a8.safetensors` (≈ 44 Go) | `<dossier modèles>/` |
| `vae/hunyuan_image_3_vae_fp16.safetensors` | `<dossier modèles>/hunyuan_image_3/vae/` |
| `clip_vision/hunyuan_image_3_instruct_distil_siglip2_so400m_naflex.safetensors` | `<dossier modèles>/hunyuan_image_3/clip_vision/` |

Le `config.json` et le `tokenizer.json` de Tencent sont livrés avec le plugin, dans `models/tencent/`.

Si vous avez déjà ces fichiers (pour ComfyUI par exemple), copiez-les ou liez-les à ces emplacements pour éviter un nouveau téléchargement.

## Utilisation

### Texte vers image

Écrivez un prompt en langage naturel, en anglais ou en chinois, puis lancez la génération. Les réglages par défaut sont ceux pour lesquels le modèle a été distillé : **8 étapes** et **Embedded Guidance 2,5**.

**Taille d'image.** Le modèle a été entraîné autour de 1 mégapixel. La liste *Resolution* propose ses 37 formats natifs, de 2048×512 à 512×2048. Les tailles personnalisées jusqu'à environ 1536×1536 fonctionnent. Au-delà, l'image se dégrade : générez en 1024 puis agrandissez avec le post-traitement de WanGP.

### Édition et fusion avec des images de référence

Dans le sélecteur **Reference Images**, choisissez :

- **First one is the image to edit** (`KI`) : la première image est celle à modifier, et la taille de sortie suit ses proportions. C'est le mode à utiliser pour l'édition.
- **Free composition** (`I`) : les images servent de références, et la taille de sortie est celle choisie dans *Resolution*.

Ajoutez **1 à 3 images** dans la fenêtre de références. Dans le prompt, désignez-les par leur ordre : « Mets le chat de la première image sur le canapé de la deuxième image ». Précisez aussi ce qui ne doit pas changer.

Chaque image est redimensionnée à environ 1 Mpx, comme le fait le pipeline de Tencent. Chaque image supplémentaire ralentit chaque étape.

**Conseil :** si une édition paraît trop saturée ou « cuite », changez de seed. Réutiliser la seed qui a produit l'image d'entrée pousse le résultat trop loin.

### Options supplémentaires (sous le prompt)

Les trois premières options ne s'affichent que lorsque des images de référence sont actives.

| Option | Effet |
|---|---|
| Reference Vision Strength | poids des tokens de la tour de vision SigLIP2 (1,0 = comportement de référence) |
| Reference Latent Strength | poids des latents VAE des références (1,0 = référence) |
| Vision Tokens Padding | complète chaque image à 1024 tokens de vision, comme le processeur de référence (par défaut). « Image patches only » reproduit l'ancien comportement du pack ComfyUI |
| Custom System Prompt | remplace le prompt système intégré du modèle. Laissez vide dans la plupart des cas |

### Mode avancé

- **General** : seed, nombre d'étapes, *Embedded Guidance Scale*. La guidance de ce modèle est un token d'entrée, pas du CFG : il n'y a pas de passe négative, donc pas de prompt négatif. Gardez 8 étapes, car d'autres valeurs s'éloignent de la trajectoire distillée.
- **LoRAs** : dossier `loras/hunyuan_image_3/`. Les LoRA s'appliquent aux couches denses (attention et expert partagé), pas aux experts routés quantifiés.
- **Post Processing / Misc.** : toutes les options standard de WanGP (upscaling, film grain, etc.).
- **Presets** : *Instruct-Distil Reference Settings* rétablit les réglages d'origine.

L'aperçu pendant le débruitage est une projection approximative (3 composantes principales du latent). Les couleurs ne sont pas réelles, mais la composition est visible.

## Profils mémoire

Utilisez un profil qui garde le transformer en RAM et le charge couche par couche (profils 2, 4 ou 5 si la RAM est juste, profil 3 si vous avez beaucoup de RAM).

Pour le VAE, WanGP choisit automatiquement le tuilage selon la VRAM (option *VAE Tiling* de la configuration). En cas de manque de mémoire, le décodage repasse automatiquement en tuiles de 512 puis de 256 pixels.

## Détails techniques

Ces informations sont utiles pour la maintenance.

**Poids W4A8.** Le format est celui de `comfy_kitchen` (`AsymW4A8Int8Layout`) : codes 4 bits Lloyd-Max, échelles fp8 par groupe de 16, échelles fp32 par ligne, rotation ConvRot (Hadamard 256).
- Les projections denses passent par le handler W4A8 natif de WanGP (`shared/qtypes/asym_w4a8_int8.py`).
- Les 64 experts de chaque couche sont stockés en banques 3D. Ils sont gérés par `models/hy3_experts.py`, qui réutilise les mêmes kernels (décodage Triton et GEMM int8) expert par expert.
- Le fichier stocke les échelles fp8 en `uint8`. Le plugin les réinterprète au chargement, dans `preprocess_w4a8_state_dict`.

**Échantillonnage.** Flow matching avec shift 3, Euler en fp32, `<timestep_r>` (meanflow) égal au sigma suivant, guidance égale à 2,5 × 1000 passée en embedding.

**Séquence.** La séquence complète est recalculée à chaque étape, sans cache KV. Le résultat est identique au pipeline de référence, car tout ce qui précède le bloc image est causal.

**Écart volontaire par rapport au pack ComfyUI.** Dans le bloc d'attention du VAE, l'aplatissement `b c f h w -> b 1 (f h w) c` suit le `rearrange` de la référence Tencent (permutation des canaux). Le pack ComfyUI faisait un simple `reshape`.

**Code dérivé de ComfyUI-HunyuanImage3.** `models/hy3_tokenizer.py` et `models/hy3_system_prompt.py` sont repris tels quels. `hy3_model.py` et `hy3_vae.py` sont adaptés de `model.py`, `pipeline.py` et `vae.py` du même pack.

## Dépannage

- **Images bruitées ou incohérentes** : revenez à 8 étapes et à une Embedded Guidance de 2,5.
- **Out of memory côté RAM** : fermez d'autres applications ou choisissez un profil mémoire qui épingle moins de RAM.
- **« at most 3 Reference Images »** : le modèle n'accepte pas plus de 3 références.
- **« needs N positions … handles 22800 »** : la séquence est trop longue. Réduisez la résolution, le nombre de références ou la longueur du prompt.
- **Le modèle n'apparaît pas** : vérifiez que le plugin est activé et que WanGP a été redémarré. La console doit afficher `[PluginManager] Loaded model plugin: HunyuanImage 3.0 Instruct-Distil`.

## Licences

- **Code du plugin** : GPL-3.0 (voir `LICENSE`), comme le pack ComfyUI dont il est dérivé (© 2026 PedroMarinhoDev, modifié pour WanGP). Des parties du modèle, du VAE et du tokenizer viennent de l'implémentation de référence de Tencent et restent soumises à la Tencent Hunyuan Community License.
- **Poids, `config.json` et `tokenizer.json`** : [Tencent Hunyuan Community License](LICENSE-TENCENT-HUNYUAN). **Cette licence ne s'applique pas dans l'Union européenne, au Royaume-Uni ni en Corée du Sud** : elle n'accorde aucun droit d'usage sur ces territoires. Elle inclut aussi une politique d'usage acceptable. Voir `NOTICE`.

## Crédits

- [Tencent Hunyuan](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0) pour HunyuanImage-3.0.
- [PedroMarinhoDev](https://github.com/PedroMarinhoDev/ComfyUI-HunyuanImage3) pour le portage ComfyUI, les poids W4A8 et les fichiers convertis.
- [Comfy-Org comfy_kitchen](https://github.com/Comfy-Org) pour le format W4A8 ConvRot.
- [DeepBeepMeep / WanGP](https://github.com/deepbeepmeep/Wan2GP) et mmgp.
