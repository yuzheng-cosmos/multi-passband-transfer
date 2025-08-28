# Multi-Passband Transfer (MPT)

This repository provides the implementation of the **Multi-Passband Transfer (MPT)** method, developed for the *Euclid* redshift calibration pipeline.  
The method statistically degrades deep photometry into wide-like observations while preserving the full correlation structure between fluxes and flux errors across multiple passbands.  

It is designed as a fast, catalogue-level alternative to image simulations such as **Balrog**, enabling the processing of millions of sources within minutes.  
MPT supports direct applications in redshift calibration for weak lensing cosmology and can also be adapted for other analyses requiring realistic wide-like mock photometry.

---

## 🔬 Scientific Context

- **Problem:** Photometric redshift ($n(z)$) calibration requires representative samples. Deep spectroscopic samples differ from wide photometric samples due to depth, noise, and survey strategy.  
- **Solution:** The MPT method statistically matches deep galaxies to their wide-like counterparts in multi-passband flux–error space, preserving correlations across bands.  
- **Validation:**  
  - Tested on DES Y3 **Balrog** data: reproduces 8D flux/error distributions with Wasserstein distance ≈ 1.02 (close to the theoretical limit of 0.61).  
  - Applied to the *Euclid* **Flagship** simulation: reduces mean-redshift biases across all tomographic bins, achieving compliance with *Euclid* requirements in up to 77% of cases up to $z=2.5$.  

For details, see the accompanying publication:  
[Y.kang et al. in prep.]

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/MPT.git
cd MPT
```

---

## 🚀 Usage

The MPT method is executed via a simple command-line interface:

```bash
python MPT.py config.yaml
```

### Configuration

All parameters are set in the `config.yaml` file.  
Typical options include:

```yaml
# Input catalogues
deep_fits: path/to/deep.fits
wide_fits: path/to/wide.fits

# Nearest-neighbour search
nn_k: 50

# Number of realisations per object
num_realisations: 10

# Output file
output_fits: wide_like.fits
```

To run with new settings, simply edit `config.yaml` and re-run the command.

---

## 📊 Output

The code produces a **wide-like catalogue** with the same objects as the input deep sample, but degraded to match the noise properties and correlations of the wide survey.  
Outputs include:  

---

## 📄 License

This project is released under the [MIT License](LICENSE).  
Please cite the associated paper when using this code in your work.  

---
