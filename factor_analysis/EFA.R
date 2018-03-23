library("psych")
library("ggplot2")
library("reshape2")

get_corrplot = function(cormat, lim = c(-1,1)){
  melted = melt(cormat)
  p = ggplot(data=melted, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white") +
    scale_fill_gradient2(low = "yellow", high = "red", mid = "white", 
                         midpoint = 0, limit = lim, space = "Lab", 
                         name="Pearson\nCorrelation") +
    scale_y_discrete(limits = rev(levels(melted$Var2))) +
    theme(axis.text.x  = element_text(angle=45, hjust=1))
  return(p)
}

full_data = read.csv("data/credit_clean.csv", sep=";")
data = full_data[, 4:29]
correlations = cor(data[, -c(3, 13, 21, 24)])

get_corrplot(correlations)
get_corrplot(solve(correlations), lim=NULL)

# Check prerequisites
KMO(data[, -c(3, 13, 21, 24)])
scree(data[, -c(3, 13, 21, 24)])
cortest.bartlett(data[, -c(3, 13, 21, 24)])
fa_raw = fa(data[, -c(3, 13, 21, 24)], 2, fm='ml')
summary(fa_raw)
fa_raw
l = fa_raw$loadings
l[abs(l) < 0.5] = NA
