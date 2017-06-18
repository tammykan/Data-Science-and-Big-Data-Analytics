library(CausalImpact)

# Read data
sum_usd_data <- read.csv("causal_impact_data/sum_usd.csv", header = F, na.strings = c('NA',''))
stock_price_data <- read.csv("causal_impact_data/stock.csv", header = F, na.strings = c('NA'), stringsAsFactors = FALSE)
y <- as.numeric(sum_usd_data[,2])
x1 <-as.numeric(stock_price_data[,2])
data <- cbind(y, x1)

dim(data)
head(data)

matplot(data, type = "l")

# Working with dates and times
time.points <- seq.Date(as.Date("2012-11-01"), by = 1, length.out = 242)
data <- zoo(cbind(y, x1), time.points)
head(data)

# Running an analysis
pre.period <- as.Date(c("2012-11-01", "2013-01-01"))
post.period <- as.Date(c("2013-01-02", "2013-06-30"))
impact <- CausalImpact(data, pre.period, post.period)

# Plotting the results
plot(impact)

# Printing a summary table
summary(impact)
summary(impact, "report")

# Adjusting the model


# Using a custom model
