################################################################################
# Project:  "End-to-End Workflows for Liquid Biopsy Biotyping Analysis Using
#             Combined MALDI MS and Machine Learning Approach"
# DOI:
#
# Author of the script:   Bc.Jaromira Pantuckova
# Contact:                534266@mail.muni.cz
#
# Purpose:
# This workflow enables transparent and reproducible processing of MALDI-TOF 
# mass spectrometry data using R. The script is thoroughly annotated with 
# commentary to support understanding of each step and allow for user-defined 
# modifications as needed.
#
# Structure:
# The workflow is divided into three main parts for clarity and modularity:
#   1) Preprocessing of mass spectra, peak alignment, feature selection, and 
#      descriptive statistics;
#   2) Unsupervised machine learning methods, primarily Principal Component 
#      Analysis (PCA);
#   3) Supervised machine learning algorithms, including:
#        - Partial Least Squares Discriminant Analysis (PLS-DA),
#        - Support Vector Machine (SVM),
#        - Random Forest (RF),
#        - Artificial Neural Networks (ANN).
#
# Note for users:
# - This script assumes your input data are in mzML format and organized in 
#   separate folders per experimental group (e.g., "HD_1", "MM_1", etc.).
# - Each subject is assumed to have multiple (e.g., 5) replicate measurements.
# - The mzML files should be named so that the subject ID and replicate number 
#   can be clearly identified (e.g., "Subject01_1.mzML", "Subject01_2.mzML", ..., 
#   "Subject02_1.mzML").
# - If your data are in a different format than mzML, you will need to load them 
#   using appropriate functions. Once imported as `MassSpectrum` objects from the 
#   MALDIquant package, the rest of the script should work as expected.
# - These filenames (without extensions) will be used as spectrum identifiers 
#   in the analysis.
# - Please adjust the working directory (`setwd()`) and the group folder list 
#   (`group_dirs`) to reflect your own data structure.
################################################################################

# ======================
# Load Required Packages
# ======================

required_packages <- c(
  "MALDIquant", "MALDIquantForeign", "MALDIrppa", "clusterSim", "gtools",
  "PROcess", "ggplot2", "reshape2", "openxlsx", "dplyr",
  "factoextra", "FactoMineR", "ropls", "caret", "tidyverse","mixOmics"
)

installed <- required_packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(required_packages[!installed])
}
lapply(required_packages, library, character.only = TRUE)

# ======================
# Set Working Directory
# ======================

# Set this to the base folder containing group subfolders with mzML files
setwd("C:/Path/To/Your/Data")

# ================================
# Define Folders with Sample Groups
# ================================
# Modify the paths below to point to your own folders with .mzML files

group_dirs <- list(
  HD_1 = "HD_1",
  MM_1 = "MM_1"
  # Add more groups if needed, e.g., MM_2 = "MM_2"
)

# ============================================
# Helper Function: Load Spectra from Folder
# ============================================
# - If your data are in a different format than mzML, you will need to load them 
#   using appropriate functions. Once imported as `MassSpectrum` objects from the 
#   MALDIquant package, the rest of the script should work as expected.

load_spectra <- function(path) {
  file_names <- dir(path, pattern = "\\.mzml$", ignore.case = TRUE)
  spectra <- importMzMl(file.path(path, file_names))
  names(spectra) <- tools::file_path_sans_ext(file_names)  # Use file names as IDs
  return(spectra)
}

# ==============================
# Load and Combine All Spectra
# ==============================
mass_spectra <- list()

for (group_name in names(group_dirs)) {
  group_path <- group_dirs[[group_name]]
  group_spectra <- load_spectra(group_path)
  # Optionally prefix spectrum names with group
  names(group_spectra) <- paste(group_name, names(group_spectra), sep = "_")
  mass_spectra <- c(mass_spectra, group_spectra)
}

# ====================================
# Save All Imported Spectra as RDS
# ====================================
# Saves the combined spectra object for downstream use
tryCatch({
  saveRDS(mass_spectra, "subject_spectra.rds")
}, error = function(e) {
  message("Error while saving spectra: ", e$message)
})

# ====================================
# Plot All Spectra to a PDF File
# ====================================
# Generates a PDF file with 4 spectra per page for visual inspection
pdf("Raw_spectra.pdf", height = 21, width = 21)
par(mfrow = c(2, 2))  # Layout: 2x2 plots per page

counter <- 0
for (subject in names(mass_spectra)) {
  plot(mass_spectra[[subject]], 
       main = paste("Spectrum for:", subject), 
       xlab = "m/z", 
       ylab = "Intensity")
  
  counter <- counter + 1
  if (counter %% 4 == 0) {
    par(mfrow = c(2, 2))  # Reset layout every 4 plots
  }
}
dev.off()

# ============================================
# Optionally Display Each Spectrum Interactively
# ============================================
# Set to TRUE if you want to view spectra one by one
interactive_mode <- FALSE

if (interactive_mode) {
  for (subject in names(mass_spectra)) {
    plot(mass_spectra[[subject]], 
         main = paste("Spectrum for:", subject), 
         xlab = "m/z", 
         ylab = "Intensity")
    readline(prompt = "Press [Enter] to continue to the next spectrum...")
  }
}


##########################################################
#### ====== MALDIquant Preprocessing Workflow  ====== ####
##########################################################

# Note for users:
# - This part of script assumes your raw data have been loaded 
#   and stored as RDS file `subject_spectra.rds`, where each subject is a list item.
# - This part of the script performs preprocessing of mass spectra, including:
#   - Trimming, transformation, smoothing, baseline correction, normalization
#   - Alignment of peaks across spectra
#   - Averaging of technical replicates per subject
#   - Peak detection with adjustable SNR thresholds, binning of peaks across spectra, 
#     filtering of rare or uninformative peaks
#   - Construction of the final feature matrix for statistical/machine learning
# - Each major preprocessing step saves its output as an RDS file so that:
#   - You can repeat or continue analysis from any step.
#   - Steps can be automated or modified individually.
# - The script includes extensive plotting to visualize changes at each step.
#   Adjust the spectrum indices (`[[n]]`) and axis ranges (`xlim`, `ylim`) 
#   to fit your own data.
################################################################################

# Load raw spectra (as pre-imported object from .mzML conversion)
mass_spectra <- readRDS("subject_spectra.rds")

# Flatten nested list of spectra
spectra <- unlist(mass_spectra, recursive = FALSE)
names(spectra) <- names(mass_spectra)

if (length(spectra) == 0) stop("No spectra found!")

# Example plot of a raw spectrum
plot(spectra[[1]], main = "Raw spectrum", 
     xlab = "m/z", ylab = "Intensity", lwd = 2)
grid()

# Save raw spectra for reproducibility
saveRDS(spectra, "spectra.rds")
# spectra <- readRDS("spectra.rds") # Load if already saved


#######################
#### Quality check ####
#######################

# Basic control to ensure input spectra are suitable for further processing.

# Check if any spectra are empty
any(sapply(spectra, isEmpty))  
# → Returns TRUE if there is at least one spectrum without data.

# Check if all spectra have regular m/z spacing (required for some algorithms)
all(sapply(spectra, isRegular))  
# → Returns TRUE if all spectra are regular.

# Identify which spectra are irregular (if any)
which(!sapply(spectra, isRegular))  
# → Returns indices of non-regular spectra, which may need to be removed or interpolated.


################################################
#### Trimming spectra to a common m/z range ####
################################################

# WARNING: The trimming range must be adapted by the user based on their own data!
# Use the visual inspection of raw spectra above to choose a relevant m/z region (range).
spectra_trimmed <- lapply(spectra, function(s) trim(s, range = c(2800, 10000)))

# Plot example trimmed spectrum
plot(spectra_trimmed[[1]], main = "Trimmed spectrum", 
     xlab = "m/z", ylab = "Intensity", lwd = 2)
grid()

# Visualize cropped part (optional)
plot(spectra[[1]], main = "Cropped high-m/z region", 
     xlab = "m/z", ylab = "Intensity", lwd = 2, xlim=c(10000, 20000), ylim=c(0, 1000))
grid()

saveRDS(spectra_trimmed, "spectra_trimmed.rds")
# spectra_trimmed <- readRDS("spectra_trimmed.rds")


# === OPTIONAL: Resampling ===
# If you want to apply resampling to get equidistant m/z axis, you can try:
# MZmin <- 2000
# MZmax <- 10000
# newMZ <- seq(from = MZmin, to = MZmax, by = 0.025)
# resampMSs <- function(spectrum) {
#   yi <- spline(x = spectrum@mass, y = spectrum@intensity, xout = newMZ, method = "fmm", ties = mean)
#   rsmp <- MALDIquant::createMassSpectrum(mass = yi$x, intensity = yi$y)
#   return(rsmp)
# }
# spectra_resampled <- lapply(spectra_trimmed, resampMSs)
# saveRDS(spectra_resampled, "spectra_resampled.rds")


#################################################
#### Intensity transformation (log and sqrt) ####
#################################################

spectra_log <- transformIntensity(spectra_trimmed, method = "log")
spectra_sqrt <- transformIntensity(spectra_trimmed, method = "sqrt")

# Visual comparison of transformations
# Choose the one that stabilizes variance while preserving peak structure or none.
par(mfrow = c(3, 1))  
plot(spectra_trimmed[[1]], main="Original", xlab = "m/z", ylab = "Intensity", lwd = 2, xlim=c(2800, 8000))
plot(spectra_log[[1]], main="Log transformed", xlab = "m/z", ylab = "Intensity", lwd = 2, xlim=c(2800, 8000))
plot(spectra_sqrt[[1]], main="Square root transformed", xlab = "m/z", ylab = "Intensity", lwd = 2, xlim=c(2800, 8000))
par(mfrow = c(1, 1))


###############################################
#### Smoothing using Savitzky-Golay filter ####
###############################################

# This step reduces noise while preserving peak shapes.
# NOTE: User must select a suitable `halfWindowSize` based on visual inspection.
#       Larger values smooth more but can also flatten peaks.
spectra_smooth10 <- smoothIntensity(spectra_trimmed, method = "SavitzkyGolay", halfWindowSize = 10)
spectra_smooth20 <- smoothIntensity(spectra_trimmed, method = "SavitzkyGolay", halfWindowSize = 20)
spectra_smooth100 <- smoothIntensity(spectra_trimmed, method = "SavitzkyGolay", halfWindowSize = 100)
spectra_smooth175 <- smoothIntensity(spectra_trimmed, method = "SavitzkyGolay", halfWindowSize = 175)

# Visual comparison 
par(mfrow = c(3, 2))
plot(spectra_trimmed[[1]], main="Before smoothing", xlim=c(3900, 4700), ylim=c(20, 370), xlab = "m/z", ylab = "Intensity", lwd = 2)
plot(spectra_smooth10[[1]], main="Smoothed (10)", xlim=c(3900, 4700), ylim=c(20, 370), xlab = "m/z", ylab = "Intensity", lwd = 2)
plot(spectra_smooth20[[1]], main="Smoothed (20)", xlim=c(3900, 4700), ylim=c(20, 370), xlab = "m/z", ylab = "Intensity", lwd = 2)
plot(spectra_smooth100[[1]], main="Smoothed (100)", xlim=c(3900, 4700), ylim=c(20, 370), xlab = "m/z", ylab = "Intensity", lwd = 2)
plot(spectra_smooth175[[1]], main="Smoothed (175)", xlim=c(3900, 4700), ylim=c(20, 370), xlab = "m/z", ylab = "Intensity", lwd = 2)

par(mfrow=c(2,1))
plot(spectra_trimmed[[1]], main="Before smoothing", ylab = "Intensity", xlim=c(3000, 10000))
plot(spectra_smooth100[[1]], main="After smoothing (100)", ylab = "Intensity", xlim=c(3000, 10000))
par(mfrow=c(1,1))

# Save the one you chose
saveRDS(spectra_smooth100, "spectra_smooth100.rds")
# spectra_smooth100 <- readRDS("spectra_smooth100.rds")


#####################################################
#### Baseline correction (comparison of methods) ####
#####################################################

# Several baseline estimation methods are available in MALDIquant.
# NOTE: User should compare methods visually and select the most appropriate one.
baseline_SNIP <- estimateBaseline(spectra_smooth100[[1]], method="SNIP", iterations=500)
baseline_TopHat <- estimateBaseline(spectra_smooth100[[1]], method="TopHat")
baseline_ConvexHull <- estimateBaseline(spectra_smooth100[[1]], method="ConvexHull")
baseline_median <- estimateBaseline(spectra_smooth100[[1]], method="median")

# Loess baseline using external function (not part of MALDIquant)
matspect <- cbind(mass=spectra_smooth100[[1]]@mass, intensity=spectra_smooth100[[30]]@intensity)
baseline_loess <- bslnoff(matspect, method="loess")
baseline_loess[,2] <- matspect[,2] - baseline_loess[,2]

# Visual comparison of baseline methods
pdf("Baseline_correction_comparison.pdf", width = 14, height = 7)
par(mfrow = c(3, 2))
plot(spectra_smooth100[[1]], xlim = c(8400, 9600), ylim=c(0, 480), main = "SNIP", xlab = "m/z", ylab = "Intensity")
lines(baseline_SNIP, col = "red", lwd = 2)
plot(spectra_smooth100[[1]], xlim = c(8400, 9600), ylim=c(0, 480), main = "TopHat", xlab = "m/z", ylab = "Intensity")
lines(baseline_TopHat, col = "green", lwd = 2)
plot(spectra_smooth100[[1]], xlim = c(8400, 9600), ylim=c(0, 480), main = "ConvexHull", xlab = "m/z", ylab = "Intensity")
lines(baseline_ConvexHull, col = "blue", lwd = 2)
plot(spectra_smooth100[[1]], xlim = c(8400, 9600), ylim=c(0, 480), main = "Median", xlab = "m/z", ylab = "Intensity")
lines(baseline_median, col = "orange", lwd = 2)
plot(spectra_smooth100[[1]], xlim = c(8400, 9600), ylim=c(0, 480), main = "Loess", xlab = "m/z", ylab = "Intensity")
lines(baseline_loess, col = "violet", lwd = 2)
dev.off()

# Apply selected baseline correction method ("SNIP")
spectra_blremoved <- removeBaseline(spectra_smooth100, method="SNIP", iteration=500)
plot(spectra_blremoved[[1]], main="After baseline removal")

saveRDS(spectra_blremoved, "spectra_blremoved.rds")
# spectra_blremoved <- readRDS("spectra_blremoved.rds")

# Final visual check of spectrum after all preprocessing so far
plot(spectra_blremoved[[1]], main = "Preprocessed spectrum", 
     xlab = "m/z", ylab = "Intensity", lwd = 2, xlim = c(2800, 5100), ylim = c(0, 700))
grid()


##################################################
#### Normalization (scaling to max intensity) ####
##################################################

scale.max <- function(x) x / max(x)
spectra_blremoved_norm <- transfIntensity(spectra_blremoved, fun = scale.max)

# Alternatively: Total Ion Current normalization
# spectra_blremoved_norm <- calibrateIntensity(spectra_blremoved, method = "TIC")

saveRDS(spectra_blremoved_norm, "spectra_blremoved_norm.rds")
# spectra_blremoved_norm <- readRDS("spectra_blremoved_norm.rds")

# Visualization of normalization 
par(mfrow = c(1, 2))
plot(spectra_blremoved[[1]], main="Before normalization", 
     xlab = "m/z", ylab = "Intensity", lwd = 2, xlim=c(3000, 10000))
plot(spectra_blremoved_norm[[1]], main="After normalization", 
     xlab = "m/z", ylab = "Intensity", lwd = 2, xlim=c(3000, 10000))
par(mfrow = c(1, 1))


#####################################################
#### Comparison of Alignment and Warping Methods ####
#####################################################

### Aligning Spectra
# Aligns spectra using specified parameters. Adjust 'halfWindowSize', 'tolerance', 
# and 'minFrequency' if needed.

spectra_aligned <- alignSpectra(
  spectra_blremoved_norm,
  halfWindowSize = 20,
  noiseMethod = "MAD",
  SNR = 2,
  tolerance = 0.002,
  warpingMethod = "lowess",
  minFrequency = 0.2
)

plot(spectra_aligned[[1]], main = "Aligned spectrum")

saveRDS(spectra_aligned, "spectra_aligned.rds")
# spectra_aligned <- readRDS("spectra_aligned.rds")

### Warping Spectra (Alternative to Alignment)
# Detect peaks first to compute warping functions

peaks <- detectPeaks(
  spectra_blremoved_norm,
  method = "MAD",
  halfWindowSize = 20,
  SNR = 15
)

# Calculate warping functions and visualize them
par(mfrow = c(2, 2))
warpingFunctions <- determineWarpingFunctions(
  peaks,
  tolerance = 20,
  plot = TRUE,
  plotInteractive = TRUE,
  minFrequency = 0.2
)

# Apply warping to spectra
warpedSpectra <- warpMassSpectra(spectra_blremoved_norm, warpingFunctions)
par(mfrow = c(1, 1))
plot(warpedSpectra[[1]], main = "Warped spectrum")

# Warp peaks accordingly
warpedPeaks <- warpMassPeaks(peaks, warpingFunctions)

saveRDS(warpedSpectra, "warpedSpectra.rds")
# warpedSpectra <- readRDS("warpedSpectra.rds")


### Visual Comparison of Both Methods (Alignment vs Warping) ###
# The user should manually choose a representative mass range (m/z interval)
# where differences in correction methods are visually apparent.

par(mfrow = c(2, 1))

plotSpectra <- function(original, corrected, method, range) {
  plot(
    original[[1]],
    main = paste0("Original spectra (", method, ", mass ", paste0(range, collapse = ":"), " Da)"),
    xlim = range, ylim = c(0, 0.8),
    type = "n", ylab = "Intensity"
  )
  color <- rainbow(length(original))
  for (i in seq(along = original)) {
    lines(original[[i]], col = color[i])
  }
  
  plot(
    corrected[[1]],
    main = paste0("Corrected spectra (", method, ", mass ", paste0(range, collapse = ":"), " Da)"),
    xlim = range, ylim = c(0, 0.8),
    type = "n", ylab = "Intensity"
  )
  for (i in seq(along = corrected)) {
    lines(corrected[[i]], col = color[i])
  }
}

# Example: Compare alignment vs original
windows()
par(mfrow = c(1, 2))
plotSpectra(spectra_blremoved_norm, spectra_aligned, "AlignSpectra", c(6400, 6500))
par(mfrow = c(1, 1))

# Example: Compare warping vs original
windows()
par(mfrow = c(1, 2))
plotSpectra(spectra_blremoved_norm, warpedSpectra, "WarpMassSpectra", c(6400, 6500))
par(mfrow = c(1, 1))


### Choosing a Reference Spectrum ###
# Function to find the most representative spectrum based on correlation

get_reference_spectrum_correlation <- function(spectra) {
  correlation_matrix <- cor(
    do.call(cbind, lapply(spectra, function(s) intensity(s))),
    use = "pairwise.complete.obs"
  )
  avg_correlation <- rowMeans(correlation_matrix, na.rm = TRUE)
  reference_index <- which.max(avg_correlation)
  return(spectra[[reference_index]])
}


# Select reference spectrum for alignment comparison
reference_spectrum <- get_reference_spectrum_correlation(spectra_blremoved_norm)
reference_spectrum

### Diagnostic Comparison of Alignment vs Warping ###

alignment_shifts <- unlist(lapply(
  spectra_aligned,
  function(s) mean(mass(s) - mass(reference_spectrum), na.rm = TRUE)
))

warping_shifts <- unlist(lapply(
  warpedSpectra,
  function(s) mean(mass(s) - mass(reference_spectrum), na.rm = TRUE)
))

# Boxplot comparison of average shifts from the reference spectrum
boxplot(alignment_shifts, warping_shifts, 
        names = c("Spectra Alignment", "Spectra Warping"),
        col = c("#1f77b4", "#d62728"),
        main = "Comparison of Spectral Shifts",
        xlab = "Correction Method",
        ylab = "Average Shift (Da)",
        cex.main = 1.5, cex.lab = 1.3, cex.axis = 1.2)

# Choose the method that gives a better result based on boxplots and visual comparison


#######################################
#### Averaging Spectra per Subject ####
#######################################

# Averages multiple replicate spectra (e.g., 5 technical replicates) into one
# final spectrum per subject. Replicates must be grouped accordingly.

# Assign original names to aligned spectra
names(spectra_aligned) <- names(mass_spectra)
names(spectra_aligned)

# Average spectra within each subject group
average_subject_spectra <- lapply(subject_groups, function(subject_files) {
  spectra_to_average <- spectra_aligned[subject_files]
  if (length(spectra_to_average) > 0) {
    averageMassSpectra(spectra_to_average)
  } else {
    NULL
  }
})

# Sort averaged spectra by subject name
average_subject_spectra <- average_subject_spectra[mixedsort(names(average_subject_spectra))]
names(average_subject_spectra)

# Save the averaged spectra
tryCatch({
  saveRDS(average_subject_spectra, "average_subject_spectra.rds")
}, error = function(e) {
  message("Error while saving averaged spectra: ", e$message)
})

# average_subject_spectra <- readRDS("average_subject_spectra.rds")


### Visualizing Averaged Spectra in PDF
# Creates a PDF with 4 averaged spectra per page for quality control

pdf("Average_spectra.pdf", height = 21, width = 21)
par(mfrow = c(2, 2))
counter <- 0
for (subject in names(average_subject_spectra)) {
  plot(average_subject_spectra[[subject]], 
       main = paste("Average Spectrum for:", subject), 
       xlab = "m/z", 
       ylab = "Intensity")
  
  counter <- counter + 1
  if (counter %% 4 == 0) {
    par(mfrow = c(2, 2))
  }
}
dev.off()


#################################
#### Feature Matrix Creation ####
#################################

# Create a metadata table indicating patient ID and health status
spectra.info <- data.frame(
  patientID = names(average_subject_spectra),
  health = ifelse(grepl("HD", names(average_subject_spectra)), "HD", "MM")
)
spectra.info

# Merge all averaged spectra into a single object (with appropriate labeling)
avg_spectra <- averageMassSpectra(average_subject_spectra , labels = spectra.info$patientID)

# Keep only unique patient entries
avgSpectra.info <- spectra.info[!duplicated(spectra.info$patientID), ]

saveRDS(avg_spectra, "avg_Spectra.rds")
# avg_spectra <- readRDS("avgspectra.rds")


#### Peak Detection ####

# Estimate noise in a representative spectrum to guide SNR threshold selection
noise <- estimateNoise(avg_spectra[[1]])

# Visualize spectrum with noise levels to decide on the appropriate SNR threshold
plot(avg_spectra[[1]], xlim = c(2800, 10000), ylim = c(0, 0.006), 
     main = "Spectrum with Estimated Noise Levels", 
     xlab = "m/z (mass-to-charge)", 
     ylab = "Intensity", col = "black", lwd = 2)
lines(noise, col = "red", lwd = 2, lty = 2)                 # SNR = 1
lines(noise[, 1], 5 * noise[, 2], col = "blue", lwd = 2, lty = 2)  # SNR = 5
lines(noise[, 1], 10 * noise[, 2], col = "green", lwd = 2, lty = 2) # SNR = 10
legend("topright", legend = c("SNR = 1", "SNR = 5", "SNR = 10"), 
       col = c("red", "blue", "green"), lty = 2, lwd = 2)

# Detect peaks in all averaged spectra using the selected SNR threshold
peaks <- lapply(avg_spectra, function(X) {
  detectPeaks(X, method = "MAD", SNR = 10, halfWindowSize = 50)
})
summary(peaks)

# Visualize detected peaks in a selected spectrum
plot(avg_spectra[[1]], xlim = c(2800, 6200), ylim = c(0, 0.05), 
     main = "Detected Peaks in Selected Spectrum", 
     xlab = "m/z (mass-to-charge)", 
     ylab = "Intensity", col = "black", lwd = 2)
points(peaks[[1]], col = "red", pch = 4, cex = 1.2, lwd = 2)
legend("topright", legend = "Detected Peaks", col = "red", pch = 4)


#### Binning and Filtering ####

# Bin peaks across spectra – peaks within 20 Da are treated as the same
peaks <- binPeaks(peaks, tolerance = 20)

# Filter out rare peaks – only retain peaks present in ≥10% of samples
peaks <- filterPeaks(peaks, minFrequency = 0.1, mergeWhitelists = TRUE)
summary(peaks)


#### Constructing the Feature Matrix ####

# Create the intensity matrix (samples × binned peaks)
featureMatrix <- as.data.frame(intensityMatrix(peaks, avg_spectra))

# Round column names to nearest whole number for better readability
colnames(featureMatrix) <- as.character(round(as.numeric(colnames(featureMatrix)), 0))

# Round intensity values to 5 decimal places
featureMatrix <- round(featureMatrix, 5)
dim(featureMatrix)

# Remove columns that contain only zeros (i.e., no intensity across all samples)
featureMatrix <- featureMatrix[, colSums(featureMatrix, na.rm = TRUE) != 0, drop = FALSE]

# Remove columns with zero variance (constant values)
featureMatrix <- featureMatrix[, apply(featureMatrix, 2, function(x) var(x, na.rm = TRUE) > 0)]
dim(featureMatrix)

# Assign sample IDs and classes as rownames and label column
rownames(featureMatrix) <- names(avg_spectra)
featureMatrix$class <- as.factor(substr(names(avg_spectra), 1, 2))  # Assumes prefix "HD"/"MM" in names

# Sort the matrix by class
featureMatrix <- featureMatrix[order(featureMatrix$class), ]

# Check for consistency in patient IDs
if (!all(names(average_subject_spectra) %in% avgSpectra.info$patientID)) {
  warning("Some spectrum names are missing from avgSpectra.info$patientID!")
}

# Export feature matrix to Excel
tryCatch({
  write.xlsx(featureMatrix, file = "DATA_MATRIX.xlsx", asTable = FALSE, colnames = TRUE, rownames = TRUE)
}, error = function(e) message("Error while saving to Excel: ", e$message))

# Save feature matrix as RDS object for downstream processing
tryCatch({
  saveRDS(featureMatrix, "feature_Matrix.rds")
}, error = function(e) message("Error while saving feature matrix: ", e$message))



#####################################################################
# === Exploratory Multivariate Analysis of Mass Spectrometry Data ===
#####################################################################

# Note for users:
# - This part of the script performs exploratory analysis of the final
#   feature matrix (peaks × samples) prior to training machine learning models.
# - Included methods: 
#     - Boxplot visualization of all features by class
#     - Principal Component Analysis (PCA)
#     - Partial Least Squares Discriminant Analysis (PLS-DA)
#     - Orthogonal PLS-DA (OPLS-DA)
# - The script also extracts the most important features (m/z values)
#   based on PCA and OPLS-DA contribution scores, and visualizes them.
# - Results such as plots and PCA coordinates are automatically saved 
#   as PDF and Excel files for further use.
# - Ensure that `feature_Matrix.rds` exists and contains a data frame 
#   with m/z features in columns and class labels in the last column.
###############################################################################

#### Data Preparation ####

featureMatrix <- readRDS("feature_Matrix.rds")

# Identify number of feature columns (last column is "class")
e <- ncol(featureMatrix)
f <- e - 1


###### Boxplots of All Features by Class #####

# Boxplot function for a single m/z variable
BOX <- function(X) {
  ggplot(featureMatrix, aes(x = class, y = .data[[X]], fill = class)) +
    geom_boxplot(outlier.colour = "red", outlier.shape = 16, outlier.size = 2, alpha = 0.7) +
    geom_jitter(color = "black", size = 1.5, alpha = 0.6, width = 0.2) +
    scale_fill_manual(values = c("#1f77b4", "#d62728")) +
    labs(title = paste("Intensity distribution for", X),
         x = "Group",
         y = "Relative intensity (0–1)") +
    theme_minimal(base_size = 18) +
    theme(axis.text.x = element_text(angle = 20, vjust = 1, hjust = 1),
          legend.position = "none")
}

# Automatically save all boxplots to a single PDF
pdf("BOX_PLOTS.pdf")
lapply(names(featureMatrix[, 1:f]), BOX)
dev.off()


######################
#### PCA Analysis ####
######################

# Filter for relevant groups (e.g. "HD" vs "MM") and ensure class is a factor
featureMatrix_TEST <- featureMatrix %>%
  filter(class %in% c("HD", "MM")) %>%
  mutate(class = as.factor(class)) %>%
  as.data.frame()

# Perform PCA
res.pca <- PCA(featureMatrix_TEST %>% select(-class))

# Extract PCA coordinates
pca_ind <- as.data.frame(res.pca$ind$coord)
pca_ind$class <- featureMatrix_TEST$class

# Plot PCA using ggplot2
ggplot(pca_ind, aes(x = Dim.1, y = Dim.2, color = class)) +
  geom_point(size = 3) +
  scale_color_manual(values = c("HD" = "blue", "MM" = "red")) +
  labs(title = "PCA Analysis", x = "PC1", y = "PC2") +
  theme_minimal()

# Plot PCA with ellipses using factoextra
fviz_pca_ind(res.pca, label = "none", habillage = featureMatrix_TEST$class, 
             palette = c("#1f77b4", "#d62728"),
             addEllipses = TRUE, pointsize = 3,
             legend.title = "Group") +
  labs(title = "PCA of Samples",
       x = "PC1", y = "PC2") +
  theme_minimal(base_size = 15)

# Scree plot of explained variance
fviz_screeplot(res.pca, addlabels = TRUE, ylim = c(0, 45)) +
  labs(title = "Explained Variance by Components",
       x = "Principal Components (PC)", 
       y = "Explained variance (%)") +
  theme_minimal(base_size = 15)


# Export PCA results (first 5 PCs + class)
PCA_EXP <- as.data.frame(res.pca$ind$coord[, 1:5])
PCA_EXP$class <- featureMatrix_TEST$class
rownames(PCA_EXP) <- rownames(featureMatrix_TEST)
write.xlsx(PCA_EXP, file = "PCA_EXPORT.xlsx", asTable = FALSE)

# Save top contributing variables to each PC
pdf("PCA_Contribution_Variables.pdf", width = 10, height = 7)
for (i in 1:3) {
  print(fviz_contrib(res.pca, choice = "var", axes = i, top = 15) +
          labs(title = paste("Top contributing variables to PC", i),
               x = "Variables (m/z)", y = "Contribution (%)") +
          theme_minimal(base_size = 16))
}
dev.off()

# Identify 3 most important m/z values based on contribution to PC1 and PC2
pca_contrib <- get_pca_var(res.pca)$contrib
important_pca_mz <- rownames(pca_contrib)[order(rowSums(pca_contrib[, 1:2]), decreasing = TRUE)[1:3]]

# Plot boxplots for the top m/z features by PCA
for (mz in important_pca_mz) {
  p <- ggplot(featureMatrix_TEST, aes(x = class, y = .data[[mz]], fill = class)) +
    geom_boxplot(outlier.colour = "red", outlier.shape = 16, outlier.size = 2, alpha = 0.7) +
    geom_jitter(color = "black", size = 1.5, alpha = 0.6, width = 0.2) +
    scale_fill_manual(values = c("#1f77b4", "#d62728")) +
    labs(title = paste("Boxplot by PCA m/z:", mz),
         x = "Group", y = "Intensity") +
    theme_minimal(base_size = 14)
  print(p)
}


############################
#### PLS-DA and OPLS-DA ####
############################

### PLS-DA analysis ###
sacurine.plsda <- plsda(select(featureMatrix_TEST, -class), featureMatrix_TEST$class)

# Extract and plot scores for first two components
scores_matrix <- sacurine.plsda$scores
scores_df <- data.frame(Comp1 = scores_matrix[, 1], Comp2 = scores_matrix[, 2],
                        class = featureMatrix_TEST$class)

ggplot(scores_df, aes(x = Comp1, y = Comp2, color = class)) +
  geom_point(size = 3) +
  labs(title = "Score Plot PLS-DA", x = "Comp 1", y = "Comp 2") +
  theme_minimal()


### OPLS-DA analysis (set number of predictive and orthogonal components) ###
sacurine.oplsda <- opls(select(featureMatrix_TEST, -class), 
                        featureMatrix_TEST$class,
                        predI = 1, orthoI = 5)

# Plot OPLS-DA scores with group labels
windows()
plot(sacurine.oplsda, typeVc = "x-score", parEllipsesL = TRUE,
     parCexN = 0.8, parLabVc = as.character(featureMatrix_TEST$class),
     parPaletteVc = c("blue", "red"))
legend("topright", legend = c("HD", "MM"), col = c("blue", "red"), pch = 15)

# Extract and visualize top 3 m/z features by VIP score (OPLS-DA)
oplsda_contrib <- sacurine.oplsda@vipVn
important_oplsda_mz <- names(sort(oplsda_contrib, decreasing = TRUE)[1:3])

for (mz in important_oplsda_mz) {
  if (mz %in% colnames(featureMatrix_TEST)) {
    p <- ggplot(featureMatrix_TEST, aes(x = class, y = .data[[mz]], fill = class)) +
      geom_boxplot(outlier.colour = "red", outlier.shape = 16, outlier.size = 2, alpha = 0.7) +
      geom_jitter(color = "black", size = 1.5, alpha = 0.6, width = 0.2) +
      scale_fill_manual(values = c("#1f77b4", "#d62728")) +
      labs(title = paste("Boxplot by OPLS-DA m/z:", mz),
           x = "Group", y = "Intensity") +
      theme_minimal(base_size = 14)
    print(p)
  } else {
    message("Value ", mz, " not found in the dataset and will not be plotted.")
  }
}


#############################################################
# === Machine Learning Workflow: Classification Models === #
#############################################################

# Note for users:
# - This section assumes you have a feature matrix saved as `feature_Matrix.rds`,
#   which includes preprocessed peak intensities and a column `class` indicating group membership.
# - This script performs classification using four supervised machine learning models:
#   Partial Least Squares Discriminant Analysis (PLS-DA), Random Forest (RF),
#   Support Vector Machines (SVM), and Artificial Neural Networks (ANN).
# - Model performance is evaluated using either:
#     * Leave-One-Out Cross-Validation (LOOCV),
#     * 10× repeated 5-fold CV,
#     * or a Train/Test split (70/30), depending on the data size.
# - You can choose the evaluation strategy by setting the `cv_strategy` parameter.
# - If your input data are not in the provided .rds format, load them using an appropriate method
#   before running this workflow. The rest of the script should remain compatible.
#################################################################################################

# === Install and load required packages ===
required_packages <- c("caret", "pROC", "randomForest", "nnet", "dplyr", 
                       "ggplot2", "glmnet", "tidyr", "reshape2", "patchwork", 
                       "pls", "kernlab", "ggpubr", "NeuralNetTools", "DiagrammeR", 
                       "openxlsx", "scales")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# === Load the feature matrix ===
featureMatrix <- readRDS("feature_Matrix.rds")
dim(featureMatrix)  # Show dimensions of the dataset


# Ensure proper data structure
featureMatrix$class <- as.factor(featureMatrix$class)

# Shuffle the rows to prevent class ordering effects
set.seed(691)
featureMatrix <- featureMatrix[sample(nrow(featureMatrix)), ]

# Optional: export to Excel for record keeping or inspection
tryCatch({
  write.xlsx(featureMatrix, file = "DATA_MATRIX2.xlsx", asTable = FALSE,
             colNames = TRUE, rowNames = TRUE)
}, error = function(e) message("Error saving Excel file: ", e$message))


#### === Train classification models === ####
# - The `train()` function from the caret package is used for model training.
# - You are encouraged to modify model-specific parameters (e.g., `tuneGrid`, `preProcess`, etc.)
#   within each `train()` call to optimize performance for your own dataset.
# - Select evaluation strategy: "LOOCV", "repeatedcv", or "split"

cv_strategy <- "LOOCV"  # Options: "LOOCV", "repeatedcv", "split"

set.seed(356)
if (cv_strategy == "LOOCV") {
  control <- trainControl(method = "LOOCV", savePredictions = "all", 
                          returnResamp = "all", classProbs = TRUE)
  training_data <- featureMatrix
  
} else if (cv_strategy == "repeatedcv") {
  control <- trainControl(method = "repeatedcv", number = 5, repeats = 10, 
                          savePredictions = "all", returnResamp = "all", classProbs = TRUE)
  training_data <- featureMatrix
  
} else if (cv_strategy == "split") {
  index <- createDataPartition(featureMatrix$class, p = 0.7, list = FALSE)
  training_data <- featureMatrix[index, ]
  test_data <- featureMatrix[-index, ]
  control <- trainControl(method = "none", classProbs = TRUE)
}

# Create an empty data frame to store accuracy results
accuracy_results <- data.frame(Model = character(), Accuracy = numeric())


# -------------------- PLS-DA --------------------
mod_pls_l <- train(class ~ ., data = training_data, method = "pls", 
                   metric = "Accuracy", trControl = control)

if (cv_strategy == "split") {
  pls_preds <- predict(mod_pls_l, newdata = test_data)
  pls_accuracy <- mean(pls_preds == test_data$class) * 100
} else {
  pls_accuracy <- mean(mod_pls_l$pred$pred == mod_pls_l$pred$obs) * 100
}

accuracy_results <- rbind(accuracy_results,
                          data.frame(Model = "PLS-DA", Accuracy = pls_accuracy))


# -------------------- Random Forest --------------------
mod_rf_l <- train(class ~ ., data = training_data, method = "rf", 
                  metric = "Accuracy", trControl = control)

if (cv_strategy == "split") {
  rf_preds <- predict(mod_rf_l, newdata = test_data)
  rf_accuracy <- mean(rf_preds == test_data$class) * 100
} else {
  rf_accuracy <- mean(mod_rf_l$pred$pred == mod_rf_l$pred$obs) * 100
}

accuracy_results <- rbind(accuracy_results,
                          data.frame(Model = "RF", Accuracy = rf_accuracy))


# -------------------- Support Vector Machine --------------------
mod_svm_l <- train(class ~ ., data = training_data, method = "svmLinear", 
                   metric = "Accuracy", trControl = control)

if (cv_strategy == "split") {
  svm_preds <- predict(mod_svm_l, newdata = test_data)
  svm_accuracy <- mean(svm_preds == test_data$class) * 100
} else {
  svm_accuracy <- mean(mod_svm_l$pred$pred == mod_svm_l$pred$obs) * 100
}

accuracy_results <- rbind(accuracy_results,
                          data.frame(Model = "SVM", Accuracy = svm_accuracy))


# -------------------- Artificial Neural Network --------------------
mod_ann_l <- train(class ~ ., data = training_data, method = "nnet",
                   metric = "Accuracy", trControl = control, trace = FALSE)

if (!is.null(mod_ann_l)) {
  if (cv_strategy == "split") {
    ann_preds <- predict(mod_ann_l, newdata = test_data)
    ann_accuracy <- mean(ann_preds == test_data$class) * 100
  } else {
    ann_accuracy <- mean(mod_ann_l$pred$pred == mod_ann_l$pred$obs) * 100
  }
  
  accuracy_results <- rbind(accuracy_results,
                            data.frame(Model = "ANN", Accuracy = ann_accuracy))
}


# === Examine optimal parameters for each model === #
mod_pls_l$bestTune         # Number of PLS components
mod_rf_l$bestTune          # mtry
mod_rf_l$finalModel$ntree  # Number of trees in RF
mod_svm_l$bestTune         # Cost parameter C
mod_ann_l$bestTune         # Size and decay
mod_ann_l$finalModel       # Neural net weights and architecture

# === Overall model accuracy results === #
print(accuracy_results)

# Set model order for plotting
accuracy_results$Model <- factor(accuracy_results$Model, levels = c("PLS-DA", "RF", "SVM", "ANN"))

# === Accuracy bar plot === #
ggplot(accuracy_results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6, alpha = 0.8) +
  theme_minimal(base_size = 14) +
  labs(title = "Overall Classification Accuracy",
       x = "Model",
       y = "Accuracy (%)") +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(breaks = seq(50, 100, 5)) +
  coord_cartesian(ylim = c(50, 100)) +
  theme(
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 13),
    plot.title = element_text(size = 18, face = "bold")
  )


# === Show per-resample accuracy and boxplot for repeated CV === #

if (cv_strategy == "repeatedcv") {
  
  # Compute accuracy for each Resample (fold × repeat) for all models
  accuracy_folds <- list(
    PLS_DA = mod_pls_l$pred %>% group_by(Resample) %>% summarise(Accuracy = mean(pred == obs) * 100),
    RF     = mod_rf_l$pred  %>% group_by(Resample) %>% summarise(Accuracy = mean(pred == obs) * 100),
    SVM    = mod_svm_l$pred %>% group_by(Resample) %>% summarise(Accuracy = mean(pred == obs) * 100)
  )
  
  # Add ANN if available
  if (!is.null(mod_ann_l)) {
    accuracy_folds$ANN <- mod_ann_l$pred %>% group_by(Resample) %>% summarise(Accuracy = mean(pred == obs) * 100)
  }
  
  # Combine all model results into one data frame for plotting
  accuracy_folds_df <- bind_rows(
    lapply(names(accuracy_folds), function(model) {
      df <- accuracy_folds[[model]]
      df$Model <- model
      df
    }),
    .id = "id"
  )
  
  # Print per-resample accuracy
  cat("Per-resample classification accuracy (repeatedcv):\n")
  print(accuracy_folds_df)
  
  # --- Boxplot: distribution of accuracy ---
  ggplot(accuracy_folds_df, aes(x = Model, y = Accuracy, fill = Model)) +
    geom_boxplot(alpha = 0.8) +
    theme_minimal(base_size = 14) +
    labs(title = "Classification Accuracy per Resample (Repeated CV)",
         x = "Model",
         y = "Accuracy (%)") +
    scale_fill_brewer(palette = "Set2") +
    theme(
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 14),
      legend.position = "none",
      plot.title = element_text(size = 18, face = "bold")
    )
  
  # --- Forest plot (mean accuracy and 95% CI) ---
  accuracy_summary <- accuracy_folds_df %>%
    group_by(Model) %>%
    summarise(
      Mean = mean(Accuracy),
      SD = sd(Accuracy),
      N = n(),
      SE = SD / sqrt(N),
      LowerCI = Mean - qt(0.975, df = N - 1) * SE,
      UpperCI = Mean + qt(0.975, df = N - 1) * SE
    )
  
  # Add model order for plotting
  accuracy_summary$Model <- factor(accuracy_summary$Model,
                                   levels = accuracy_summary$Model[order(accuracy_summary$Mean)])
  
  ggplot(accuracy_summary, aes(x = Mean, y = Model)) +
    geom_point(size = 3) +
    geom_errorbarh(aes(xmin = LowerCI, xmax = UpperCI), height = 0.2) +
    geom_text(aes(label = sprintf("Accuracy = %.2f\n95%% CI: [%.2f, %.2f]", Mean / 100, LowerCI / 100, UpperCI / 100)),
              hjust = -0.05, color = "red", size = 4) +
    labs(title = "Comparative Accuracy of ML Models",
         x = "Accuracy", y = NULL) +
    xlim(90, 100) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(size = 18, face = "bold"),
      axis.text.y = element_text(size = 14),
      axis.text.x = element_text(size = 12)
    )
}

######################################
#### Calculation of other metrics ####
######################################

# === Function to calculate metrics from CV predictions === #
calculate_metrics_cv <- function(predictions_df) {
  actuals <- predictions_df$obs
  predicted <- predictions_df$pred
  
  # Confusion matrix and basic metrics
  conf_matrix <- confusionMatrix(predicted, actuals)
  sensitivity <- conf_matrix$byClass["Sensitivity"]
  specificity <- conf_matrix$byClass["Specificity"]
  precision <- conf_matrix$byClass["Precision"]
  accuracy <- conf_matrix$overall["Accuracy"]
  
  # F1 and F2 score calculation
  if (!is.na(precision) && !is.na(sensitivity) && (precision + sensitivity) > 0) {
    F1 <- 2 * (precision * sensitivity) / (precision + sensitivity)
    F2 <- 5 * (precision * sensitivity) / (4 * precision + sensitivity)
  } else {
    F1 <- NA
    F2 <- NA
  }
  
  # AUC calculation
  roc_curve <- roc(actuals, as.numeric(predicted), quiet = TRUE)
  auc_value <- auc(roc_curve)
  
  return(data.frame(Sensitivity = sensitivity,
                    Specificity = specificity,
                    Precision = precision,
                    Accuracy = accuracy,
                    F1 = F1,
                    F2 = F2,
                    AUC = auc_value))
}

# Compute metrics for each model (if not using train/test split)
if (cv_strategy != "split") {
  metrics_pls <- calculate_metrics_cv(mod_pls_l$pred)
  metrics_rf <- calculate_metrics_cv(mod_rf_l$pred)
  metrics_svm <- calculate_metrics_cv(mod_svm_l$pred)
  if (!is.null(mod_ann_l)) {
    metrics_ann <- calculate_metrics_cv(mod_ann_l$pred)
  }
  
  # === Create summary table of all performance metrics including F1 and F2 ===
  metrics_table <- data.frame(
    Model = c("PLS-DA", "RF", "SVM", if (!is.null(mod_ann_l)) "ANN" else NA),
    Sensitivity = c(metrics_pls$Sensitivity, metrics_rf$Sensitivity, metrics_svm$Sensitivity, if (!is.null(mod_ann_l)) metrics_ann$Sensitivity else NA),
    Specificity = c(metrics_pls$Specificity, metrics_rf$Specificity, metrics_svm$Specificity, if (!is.null(mod_ann_l)) metrics_ann$Specificity else NA),
    Precision = c(metrics_pls$Precision, metrics_rf$Precision, metrics_svm$Precision, if (!is.null(mod_ann_l)) metrics_ann$Precision else NA),
    Accuracy = c(metrics_pls$Accuracy, metrics_rf$Accuracy, metrics_svm$Accuracy, if (!is.null(mod_ann_l)) metrics_ann$Accuracy else NA),
    F1 = c(metrics_pls$F1, metrics_rf$F1, metrics_svm$F1, if (!is.null(mod_ann_l)) metrics_ann$F1 else NA),
    F2 = c(metrics_pls$F2, metrics_rf$F2, metrics_svm$F2, if (!is.null(mod_ann_l)) metrics_ann$F2 else NA),
    AUC = c(metrics_pls$AUC, metrics_rf$AUC, metrics_svm$AUC, if (!is.null(mod_ann_l)) metrics_ann$AUC else NA)
  )
  
  # === Display the summary table === #
  print(metrics_table)
} else {
  message("Detailed metrics (Sensitivity, Specificity, AUC) not computed for train/test split mode.")
}

# === Visualization of Classification Metrics per Model === #

# === Convert performance table to long format ===
# Reshape table so each metric is in its own row, which is required by ggplot
performance_long <- metrics_table %>%
  pivot_longer(cols = c(Sensitivity, Specificity, Accuracy, AUC),
               names_to = "Metric",
               values_to = "Value")

# === Barplot: All metrics across models ===
ggplot(performance_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Model Performance by Metric",
       y = "Metric Value",
       x = "Model") +
  scale_y_continuous(breaks = seq(0.5, 1, by = 0.1)) + 
  coord_cartesian(ylim = c(0.5, 1)) +
  theme_minimal()

# === Transform values for individual plots ===
# Convert metrics to percentages where appropriate
performance_long_transformed <- performance_long %>%
  filter(!is.na(Value)) %>%
  mutate(
    Value = ifelse(Metric %in% c("Sensitivity", "Specificity", "Accuracy"),
                   Value * 100, Value)
  )


# Get unique metric names
metrics_unique <- unique(performance_long_transformed$Metric)

# Create an empty list to store the plots
plot_list <- list()

# === Create individual barplots for each metric ===
for (metric in metrics_unique) {
  df_plot <- filter(performance_long_transformed, Metric == metric)
  
  p <- ggplot(df_plot, aes(x = Model, y = Value, fill = Model)) +
    geom_bar(stat = "identity", width = 0.7) +
    labs(
      title = metric,
      x = "Model",
      y = ifelse(metric == "AUC", "AUC", paste0(metric, " (%)"))
    ) +
    scale_fill_brewer(palette = "Set2") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(size = 18, face = "bold"),
      axis.title = element_text(size = 16, face = "bold"),
      axis.text.x = element_text(size = 14, face = "bold"),
      axis.text.y = element_text(size = 14),
      legend.position = "none"
    ) +
    {
      if (metric == "AUC") coord_cartesian(ylim = c(0.5, 1))
      else coord_cartesian(ylim = c(50, 100))
    }
  
  plot_list[[metric]] <- p
}

# === Combine all individual metric plots horizontally ===
combined_plot <- wrap_plots(plot_list, nrow = 1)

# Display the final figure
print(combined_plot)


#### === Variable Importance Visualization and Interpretation === ####

# This section visualizes the most important m/z variables selected by each model.
# Users can inspect which m/z values contributed most to classification performance.

# --- Variable importance from PLS-DA model ---
PLS_DA_importance <- varImp(mod_pls_l, scale = TRUE)
plot(PLS_DA_importance, top = 10, main = "Variable Importance in PLS-DA")

# --- Variable importance from Random Forest model (basic plot) ---
RF_importance <- varImpPlot(mod_rf_l$finalModel, main = "Variable Importance in Random Forest")

# --- Variable importance from SVM model ---
SVM_importance <- varImp(mod_svm_l, scale = TRUE)
plot(SVM_importance, top = 10, main = "Variable Importance in SVM")

# --- Variable importance from ANN model ---
ANN_importance <- varImp(mod_ann_l, scale = TRUE)
plot(ANN_importance, top = 10, main = "Variable Importance in ANN")


# --- Improved barplot visualization for Random Forest importance using ggplot2 ---
# This visualization provides a cleaner and more customizable plot
RF_importance <- varImp(mod_rf_l, scale = TRUE)
ggplot(RF_importance, aes(x = reorder(Overall, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  theme_minimal(base_size = 14) +
  labs(title = "Variable Importance in Random Forest",
       x = "m/z value",
       y = "Importance")


### === Aggregated Variable Importance Across All Models === ###
# This section aggregates the top 10 most important variables from each model.
# Users can compare which m/z values are consistently ranked as important across models.

var_imp_list <- list(
  PLS = varImp(mod_pls_l, scale = TRUE),
  RF = varImp(mod_rf_l, scale = TRUE),
  SVM = varImp(mod_svm_l, scale = TRUE),
  ANN = if (!is.null(mod_ann_l)) varImp(mod_ann_l, scale = TRUE) else NULL
)

# Inspect the structure (optional)
str(var_imp_list)

# Combine top 10 variables from each model into a unified dataframe
var_imp_df <- bind_rows(
  lapply(names(var_imp_list), function(model_name) {
    x <- var_imp_list[[model_name]]
    if (!is.null(x)) {
      imp <- as.data.frame(x$importance)
      imp$Variable <- rownames(imp)
      imp$Model <- model_name
      
      # Ensure "Overall" column exists (if not, compute average of numeric columns)
      if (!"Overall" %in% colnames(imp)) {
        imp <- imp %>%
          mutate(Overall = rowMeans(select(., where(is.numeric)), na.rm = TRUE))
      }
      
      imp %>% top_n(10, wt = Overall)
    } else {
      NULL
    }
  })
)

# Plot aggregated variable importance
var_imp_plot <- ggplot(var_imp_df, aes(x = reorder(Variable, Overall), y = Overall, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  coord_flip() +
  labs(title = "Aggregated Variable Importance Across Models",
       x = "m/z value",
       y = "Importance") +
  theme_minimal()

print(var_imp_plot)


### === Boxplots for Top 3 m/z Values According to ANN === ###
# This section visualizes the top 3 variables (m/z values) from the best-performing model (ANN)
# using boxplots to show group-level differences (e.g. HD vs MM).

# Extract top 3 m/z values by importance from ANN
important_mz <- rownames(ANN_importance$importance)[order(ANN_importance$importance$Overall, decreasing = TRUE)[1:3]]
important_mz <- gsub("`", "", important_mz)  # Remove backticks from variable names if present

# Reshape the data to long format for plotting
featureMatrix_long <- featureMatrix[, important_mz]
featureMatrix_long$class <- featureMatrix$class
featureMatrix_long_melt <- melt(featureMatrix_long, id.vars = "class")
names(featureMatrix_long_melt) <- c("class", "mz", "intensity")

# Create boxplots with significance testing (Wilcoxon test)
ggplot(featureMatrix_long_melt, aes(x = class, y = intensity, fill = class)) +
  geom_boxplot(outlier.shape = 16, outlier.size = 2, alpha = 0.7) +
  geom_jitter(color = "black", size = 1.5, alpha = 0.6, width = 0.2) +
  scale_fill_manual(values = c("blue", "red")) +
  facet_wrap(~ mz, scales = "free") +
  labs(title = "Boxplots for Top 3 m/z Values by ANN",
       x = "Group", y = "Intensity") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none") +
  stat_compare_means(
    method = "wilcox.test", 
    label = "p.signif", 
    comparisons = list(c("HD", "MM"))
  )


### === Wilcoxon Tests for Group Differences === ###
# This section performs Wilcoxon rank-sum tests for each of the top 3 m/z values
# to assess statistical significance of group differences (non-parametric).

# Perform Wilcoxon tests
wilcox_results_top3 <- sapply(important_mz, function(mz) {
  wilcox.test(featureMatrix[[mz]] ~ featureMatrix$class)$p.value
})

# Assign variable names to results
names(wilcox_results_top3) <- important_mz

# Print p-values
wilcox_results_top3


################################################################
# === Model Overfitting Verification and Feature Selection === #
################################################################

# This part of the script performs:
# - A permutation test to verify whether the ANN model is overfitting.
# - Feature selection based on the Wilcoxon test due to non-normal data distribution.
# - Model evaluation on filtered features.
# - Histograms for visual verification of non-normal distribution.

# ========================================
# Permutation Test for the Best Model (ANN)
# ========================================

# This section evaluates if the ANN classifier is overfitting by training it on permuted class labels.
# The achieved accuracies and AUCs on permuted labels are compared to results on real labels.
# You can change the cross-validation method below if LOOCV is not used in your workflow.

control <- trainControl(method = "LOOCV", savePredictions = "all", returnResamp = "all", classProbs = TRUE) # e.g., use repeatedcv" for repeated k-fold

n_permutations <- 100
accuracy_values <- numeric(n_permutations)
auc_values <- numeric(n_permutations)

set.seed(428)

for (i in 1:n_permutations) {
  shuffled_featureMatrix <- featureMatrix
  shuffled_featureMatrix$class <- sample(shuffled_featureMatrix$class)  # Randomly permute class labels
  
  model_shuffled <- train(class ~ ., data = shuffled_featureMatrix, method = "nnet",
                          metric = "Accuracy", trControl = control, trace = FALSE)
  
  accuracy_values[i] <- sum(model_shuffled$pred$pred == model_shuffled$pred$obs) / 
    length(model_shuffled$pred$pred)
  
  auc_values[i] <- calculate_metrics_cv(model_shuffled$pred)$AUC
}

# Plot histogram of model accuracies from permuted data
hist_accuracy <- ggplot(data.frame(Accuracy = accuracy_values), aes(x = Accuracy)) +
  geom_histogram(binwidth = 0.01, fill = "navy", color = "black") +
  geom_vline(xintercept = 1 / length(unique(featureMatrix$class)), color = "red", linetype = "dashed", size = 1) +
  labs(x = "Accuracy", y = "Frequency") +
  scale_x_continuous(limits = c(0.3, 0.7)) +
  theme_minimal(base_size = 14)

# Plot histogram of AUC values from permuted data
hist_auc <- ggplot(data.frame(AUC = auc_values), aes(x = AUC)) +
  geom_histogram(binwidth = 0.01, fill = "navy", color = "black") +
  geom_vline(xintercept = 0.5, color = "red", linetype = "dashed", size = 1) +
  labs(x = "AUC", y = "Frequency") +
  scale_x_continuous(limits = c(0.3, 0.7)) +
  theme_minimal(base_size = 14)

# Display both histograms side by side
hist_accuracy + hist_auc


# =====================================
# Feature Selection Using Wilcoxon Test
# =====================================

# Since the data is not normally distributed (see histogram plots later), 
# the Wilcoxon test is used to identify the most significant m/z features.

wilcox_results <- apply(featureMatrix[, -ncol(featureMatrix)], 2, function(x) wilcox.test(x ~ featureMatrix$class)$p.value)

# Sort features by p-value (ascending)
sorted_features <- sort(wilcox_results)

# Select top 5 most significant features (you may choose more if needed)
num_features <- min(5, length(sorted_features))
top_features <- names(sorted_features)[1:num_features]

# Create a new data matrix with selected features
filtered_wilcox_featureMatrix <- featureMatrix[, c(top_features, "class")]

# Train ANN on filtered features
model_ann_filtered <- train(class ~ ., filtered_wilcox_featureMatrix, method = "nnet",
                            metric = "Accuracy", trControl = control)

# Calculate accuracy
ann_accuracy_filtered <- sum(model_ann_filtered$pred$pred == model_ann_filtered$pred$obs) /
  length(model_ann_filtered$pred$pred) * 100

# Print accuracy
ann_accuracy_filtered

# =============================
# Boxplots of Selected Features
# =============================

# Visualize distributions of selected features between classes using boxplots
long_data <- melt(filtered_wilcox_featureMatrix, id.vars = "class")

ggplot(long_data, aes(x = class, y = value, fill = class)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~variable, scales = "free") +
  labs(title = "Boxplots of Selected Features", x = "Class", y = "Value") +
  theme_minimal()


# ======================================================
# Visual Verification of Distribution: Histogram Example
# ======================================================

# This section visually verifies that feature distributions are not normal,
# justifying the use of the Wilcoxon test instead of parametric tests.

# Choose the most significant feature
top_feature <- names(sorted_features)[1]

# Extract m/z and intensities from example spectrum
spectrum <- average_subject_spectra[[3]]
mz_vals <- mass(spectrum)
intensities <- intensity(spectrum)

# Define region of interest around the top m/z
mz_min <- 4370
mz_max <- 4450
subset_idx <- which(mz_vals >= mz_min & mz_vals <= mz_max)

# Plot spectral region around selected m/z
plot(mz_vals[subset_idx], intensities[subset_idx],
     type = "l", lwd = 2,
     main = "Zoomed Region Around m/z ~4411",
     xlab = "m/z", ylab = "Intensity")

# Create histogram plots with fitted normal curves for both classes
df_histogram <- data.frame(
  Value = featureMatrix[[top_feature]],
  Class = featureMatrix$class
)

group1 <- unique(df_histogram$Class)[1]
group2 <- unique(df_histogram$Class)[2]

# Data for each group
data_group1 <- df_histogram[df_histogram$Class == group1, ]
data_group2 <- df_histogram[df_histogram$Class == group2, ]

# Histogram + normal curve for group 1
mean1 <- mean(data_group1$Value)
sd1 <- sd(data_group1$Value)
x_vals1 <- seq(mean1 - 4 * sd1, mean1 + 4 * sd1, length.out = 100)
df_curve1 <- data.frame(x = x_vals1, y = dnorm(x_vals1, mean = mean1, sd = sd1))

plot1 <- ggplot(data_group1, aes(x = Value)) +
  geom_histogram(aes(y = ..density..), bins = 20, fill = "skyblue", alpha = 0.5, color = "black") +
  geom_line(data = df_curve1, aes(x = x, y = y), color = "red", size = 1) +
  labs(title = paste("Histogram for Group:", group1), x = "Value", y = "Density") +
  theme_minimal(base_size = 14)

# Histogram + normal curve for group 2
mean2 <- mean(data_group2$Value)
sd2 <- sd(data_group2$Value)
x_vals2 <- seq(mean2 - 4 * sd2, mean2 + 4 * sd2, length.out = 100)
df_curve2 <- data.frame(x = x_vals2, y = dnorm(x_vals2, mean = mean2, sd = sd2))

plot2 <- ggplot(data_group2, aes(x = Value)) +
  geom_histogram(aes(y = ..density..), bins = 20, fill = "skyblue", alpha = 0.5, color = "black") +
  geom_line(data = df_curve2, aes(x = x, y = y), color = "red", size = 1) +
  labs(title = paste("Histogram for Group:", group2), x = "Value", y = "Density") +
  theme_minimal(base_size = 14)

# Display both histograms side by side
plot1 + plot2  # requires patchwork

