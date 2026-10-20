#generics s

#' Extract top-ranked features from an inference result
#'
#' Returns the highest-ranked features according to significance and effect
#' size statistics from an SSANOVA-SVD inference object.
#'
#' @param x An object of class `"svdxInference"`,
#'   `"isvdxInference"`, `"svdxWorkflow"`, or `"isvdxWorkflow"`.
#' @param dataset Dataset name or index for integrated objects.
#'   Ignored for single-dataset objects.
#' @param scope Character string indicating whether rankings should be based on
#'   `"union"` statistics or a specific `"component"`.
#' @param component Component identifier when `scope = "component"`.
#' @param n Number of features to return.
#'
#' @return
#' A tibble containing the top-ranked features.
#'
#' @details
#' For integrated objects, a dataset must be specified.
#'
#' @examples
#' \dontrun{
#' topFeatures(inf, n = 20)
#'
#' topFeatures(
#'   iinf,
#'   dataset = "rna",
#'   n = 20
#' )
#' }
#'
#' @export
topFeatures <- function(x, ...) {
  UseMethod("topFeatures")
}

#' Extract significant features
#'
#' Returns features declared significant after multiple-testing correction.
#'
#' @param x An inference or workflow object.
#' @param dataset Dataset name or index for integrated objects.
#'   Ignored for single-dataset objects.
#' @param alpha Significance threshold.
#' @param scope Character string indicating whether significance should be
#'   determined from `"union"` or `"component"` statistics.
#' @param component Component identifier when `scope = "component"`.
#'
#' @return
#' Character vector of significant feature names.
#'
#' @details
#' Union significance is based on the minimum component-specific p-value with
#' Bonferroni correction across estimated components, followed by
#' Benjamini-Hochberg correction across features. This answers whether a feature
#' is associated with at least one component. Omnibus/global p-values, when
#' available in the inference object, answer a related but different question:
#' whether the latent component set jointly explains the feature.
#'
#' @examples
#' \dontrun{
#' sig <- svdxSignificant(inf)
#'
#' sig_rna <- svdxSignificant(
#'   iinf,
#'   dataset = "rna"
#' )
#' }
#'
#' @export
svdxSignificant <- function(x, ...) {
  UseMethod("svdxSignificant")
}

#' Extract the enrichment universe
#'
#' Returns the set of features that were eligible for testing and enrichment.
#'
#' @param x An inference or workflow object.
#' @param dataset Dataset name or index for integrated objects.
#'   Ignored for single-dataset objects.
#'
#' @return
#' Character vector of feature identifiers.
#'
#' @details
#' The universe is typically used as the background set for pathway
#' enrichment analyses.
#'
#' @examples
#' \dontrun{
#' universe <- svdxUniverse(inf)
#'
#' universe_rna <- svdxUniverse(
#'   iinf,
#'   dataset = "rna"
#' )
#' }
#'
#' @export
svdxUniverse <- function(x, ...) {
  UseMethod("svdxUniverse")
}

#' Convert inference statistics to a ranked feature list
#'
#' Produces a named numeric vector suitable for rank-based enrichment methods
#' such as FGSEA and GSEA.
#'
#' @param x An inference or workflow object.
#' @param dataset Dataset name or index for integrated objects.
#'   Ignored for single-dataset objects.
#' @param score Ranking statistic to use. Union-level choices include
#'   `"union_q"`, `"union_p"`, `"global_q"`, `"global_p"`,
#'   `"max_abs_stat"`, and related transformed scores. Component-level choices
#'   include `"qval"`, `"pval"`, and signed or absolute component statistics.
#' @param scope Character string indicating whether rankings should be computed
#'   from `"union"` or `"component"` statistics.
#' @param component Component identifier when `scope = "component"`.
#'
#' @return
#' Named numeric vector ordered from strongest to weakest evidence.
#'
#' @details
#' This function is intended as a bridge between SSANOVA-SVD inference results
#' and rank-based pathway enrichment methods. `union_p`/`union_q` prioritize
#' features with strong evidence in at least one component, whereas
#' `global_p`/`global_q` prioritize features whose variation is jointly
#' explained by the full component set.
#'
#' @examples
#' \dontrun{
#' ranks <- asRankedList(inf)
#'
#' ranks_rna <- asRankedList(
#'   iinf, 
#'   dataset = "rna"
#' )
#' }
#'
#' @export
asRankedList <- function(x, ...) {
  UseMethod("asRankedList")
}

#' Conditional inference for feature loadings
#'
#' Performs conditional inference for feature loadings using held-out or external
#' samples. Component scores for the test samples are predicted from metadata
#' only, and each feature is then regressed on the predicted component scores
#' and optional nuisance covariates.
#'
#' @param fit A fitted `"SSANOVASVD"` or `"iSSANOVASVD"` object.
#' @param ... Additional arguments passed to the class-specific method.
#'
#' @return For `"SSANOVASVD"` fits, an object of class `"svdxInference"`.
#'   For `"iSSANOVASVD"` fits, an object of class `"isvdxInference"` containing
#'   one `"svdxInference"`-like result per dataset.
#'
#' @details
#' The key design principle is to avoid double dipping: test-sample component
#' scores are predicted using metadata, not estimated from `Ytest`. Inference is
#' then performed by regressing held-out features on the predicted scores and
#' nuisance design matrix.
#'
#' For integrated fits, `Ytest`, `newzdata`, and `zformula` are list-valued
#' with one entry per dataset. `family` may be `NULL`, a scalar recycled to all
#' datasets, an unnamed vector/list parallel to `Ytest`, or a named vector/list
#' matched by dataset name.
#'
#' The returned union-level table includes several complementary feature-level
#' statistics. Component-specific p-values test association with individual
#' components. `union_p` is the Bonferroni-corrected minimum component p-value
#' and asks whether at least one component is associated with the feature;
#' `union_q` is the BH-adjusted version across features. `global_p` is an
#' omnibus p-value testing whether the full component set jointly explains the
#' feature. For Gaussian models this is limma's moderated F-test over the
#' component coefficients. For Poisson GLMs it is a likelihood-ratio omnibus test. For quasi-Poisson GLMs it is an analysis-of-deviance F-test scaled by the estimated dispersion. `global_q` is the BH-adjusted global p-value across features.
#'
#' @examples
#' \dontrun{
#' inf <- inferV(
#'   fit,
#'   Ytest = Y_test,
#'   newdata = sample_data_test,
#'   newzdata = sample_data_test,
#'   zformula = ~ batch
#' )
#'
#' summary(inf)
#' topFeatures(inf)
#'
#' iinf <- inferV(
#'   ifit,
#'   Ytest = Y_test_list,
#'   newdata = sample_data_test,
#'   newzdata = zdata_test_list,
#'   zformula = zformula_list
#' )
#'
#' summary(iinf)
#' topFeatures(iinf, dataset = "rna")
#' }
#'
#' @export
inferV <- function(fit, ...) {
  UseMethod("inferV")
}

#' Plot fitted component scores as functions of metadata
#'
#' Visualizes one fitted SSANOVA-SVD component as a function of one or more
#' metadata variables. Predictions are obtained from the fitted score function
#' and may be displayed either on observed samples or on a prediction grid.
#'
#' @param fit A fitted SSANOVASVD or iSSANOVASVD object.
#' @param data Metadata data frame.
#' @param component Component to visualize.
#' @param x Variable shown on the x-axis.
#' @param group Optional grouping variable.
#' @param facet Optional faceting variable.
#' @param include Optional subset of xformula terms.
#'
#' @details
#' For integrated SSANOVA-SVD models (`iSSANOVASVD`),
#' component score functions are shared across datasets.
#' Therefore no dataset argument is required.
#' 
#' @return A ggplot object.
#'
#' @export
plotU <- function(fit, ...) {
  UseMethod("plotU")
}

#' Plot xformula-term contributions to a component
#'
#' Decomposes a fitted component into individual xformula term contributions
#' and visualizes each contribution separately.
#'
#' @param fit A fitted SSANOVASVD or iSSANOVASVD object.
#' @param data Metadata data frame.
#' @param component Component to visualize.
#' @param x Variable shown on the x-axis.
#' @param group Optional grouping variable.
#'
#' @return A ggplot object.
#'
#' @details
#' Useful for understanding which terms contribute to a fitted component.
#' For integrated SSANOVA-SVD models (`iSSANOVASVD`),
#' component score functions are shared across datasets.
#' Therefore no dataset argument is required.
#' @export
plotUTerms <- function(fit, ...) {
  UseMethod("plotUTerms")
}

#' Plot a low-dimensional embedding of fitted component scores
#'
#' Displays samples in the space of two fitted components.
#'
#' @param fit A fitted SSANOVASVD or iSSANOVASVD object.
#' @param data Metadata data frame.
#' @param components Length-two vector identifying the components to plot.
#' @param color Optional coloring variable.
#' @param group Optional grouping variable.
#' @param facet Optional faceting variable.
#'
#' @return A ggplot object.
#'
#' @details
#' Requires at least two fitted components.
#' For integrated SSANOVA-SVD models (`iSSANOVASVD`),
#' component score functions are shared across datasets.
#' Therefore no dataset argument is required.
#'
#' @export
plotUEmbedding <- function(fit, ...) {
  UseMethod("plotUEmbedding")
}


#' Run over-representation enrichment for significant features
#'
#' Performs Fisher exact-test over-representation analysis for features selected
#' by an SSANOVA-SVD inference object or workflow object.
#'
#' @param inf An `"svdxInference"`, `"isvdxInference"`, `"svdxWorkflow"`, or
#'   `"isvdxWorkflow"` object.
#' @param ... Additional arguments passed to class-specific methods.
#'
#' @return A tibble with one row per feature set/pathway, including set size,
#'   number of significant features, overlap, odds ratio, p-value, and
#'   Benjamini-Hochberg adjusted q-value.
#'
#' @details
#' For single-dataset inference and workflow objects, enrichment is run directly
#' using the features returned by [svdxSignificant()] and the universe returned
#' by [svdxUniverse()].
#'
#' For integrated inference and workflow objects, supply `dataset` to select the
#' dataset on which enrichment should be performed.
#'
#' @examples
#' \dontrun{
#' feature_sets <- list(
#'   pathway_A = c("gene1", "gene2", "gene3"),
#'   pathway_B = c("gene4", "gene5", "gene6")
#' )
#'
#' enr <- enrichPathways(
#'   inf,
#'   feature_sets = feature_sets,
#'   alpha = 0.05,
#'   scope = "union"
#' )
#'
#' ienr <- enrichPathways(
#'   iinf,
#'   dataset = "rna",
#'   feature_sets = feature_sets,
#'   alpha = 0.05,
#'   scope = "union"
#' )
#' }
#'
#' @export
enrichPathways <- function(inf, ...) {
  UseMethod("enrichPathways")
}
