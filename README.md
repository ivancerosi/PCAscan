# PCAscan
Scan SQL table with PCA algorithm to determine data compression potential


## Summary:

Principal Component Analysis is a dimensionality reduction method which is applied in host of different areas ranging from scientific (investigation of latent factors/causes underlying the data) to practical (saving space with compression).
Key feature of PCA is that it transforms the data into a new space in which vector bases are orthonogal. Orthogonality means that features(columns) share no overlapping information and thus there is no redundancy. In practice however, data is usually correlated to a degree which means that data features(columns) contain overlapping information. Storing data in such form is subomptimal. PC analysis removes this redundancy by rotating the data vector bases. Data transformation is linear which means that it is possible to reconstruct original data at any point. However, PCA is done under the assumption that weaker components will be discarded to save space. Data reconstruction is still possible even with incomplete set of transformed components but there will be a certain degree of information loss. Goal of this script is to assess the degree of information loss should certain number of components be discarded after the transformation.

Principal Component Analysis algorithm is implemented via QR matrix factorization. Second element of the script is MySQL interface which can directly convert tables with numerical data into NumPy matrix viable for PCA analysis.

## How to use:

Check example.py
