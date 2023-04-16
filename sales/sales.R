library(fpp3)
library(seasonal)
library(ggplot2)
library(gridExtra)
library(dlookr)
library(progressr)
library(future)

sales <- readr::read_csv('D:/contest/kaggle/sales/train.csv')
test <- readr::read_csv('D:/contest/kaggle/sales/test.csv')

# We have to make a forecast for the next half a month
test <- test |>
  as_tsibble(index=date, key=c(store_nbr, family))

# I.
# convert sales to tsibble
sales <- sales |>
  as_tsibble(index = date, key = c(store_nbr, family) )

# number of missing values (explicit missing observations) for each column
sapply(sales, function(x) sum(is.na(x)))

# number of implicit missing observations (gaps)
sales_gaps <- sales |>
  count_gaps(.full = TRUE)

# get a better understanding
# : one day missing from each year.
sales_gaps |>
  group_by(.from) |>
  summarise(count = sum(.n))

sales |>
  filter((month(date) == 12) & (day(date) == 25))

sales <- sales |>
  fill_gaps(sales = median(sales), onpromotion = median(onpromotion), .full = TRUE)

sales <- sales |>
  fill_gaps(.full = TRUE)

# II. ignore (store_nbr, family); yield prediction
# group
sales_grouped <- sales |>
  index_by(date) |>
  summarise(total_sales=sum(sales))
?index_by

# fill_gaps left NA's
sales_grouped |>
  filter((month(date) == 12) & (day(date) == 25))

# fill the NA's with the median
sales_grouped <- sales_grouped |>
  replace_na(list(total_sales=median(sales_grouped$total_sales, na.rm = TRUE)))

# plot moving averages
sales_grouped |>
  mutate(`90-MA` = slider::slide_dbl(total_sales, mean, .before = 89, .after = 0, .complete = TRUE)) |>
  ggplot(aes(x = date)) +
  geom_line(aes(y = total_sales), color = 'grey') +
  geom_line(aes(y = `90-MA`), color = 'orange')


##########
# Try applying some natural transformations to obtain a stationary process 
# on the aggregated sales data
##########
min_max_scale <- function(data) {
  (data - min(data)) / (max(data) - min(data))
}

impute_outlier <- function(data) {
  data[data %in% boxplot(data)$out] <- median(data)
  data
}

preprocessed_sales <- sales_grouped |>
  mutate(
    log_total_sales = impute_outlier(log(total_sales))
  ) |>
  model(
    stl = STL(log_total_sales ~ trend() +
          season(),
        robust = TRUE),
    classical = classical_decomposition(log_total_sales, type=c("additive"))
  ) |>
  components() |>
  select(season_adjust) |>
  mutate(
    log_seasadj_scaled_sales = min_max_scale(season_adjust)
  )

# doesn't work as expected 
#print(preprocessed_sales |>
#  as_tibble() |>
#  pivot_wider(names_from = .model, values_from = log_seasadj_scaled_sales),
#  n = 100
#)
# workaround:
p1 <- preprocessed_sales |>
  filter(.model == "classical")
p2 <- preprocessed_sales |>
  filter(.model == "stl")
preprocessed_sales <- p1 |>
  left_join(p2, by="date", suffix = c(".classical", ".stl"))

# compare decompositions; look for the result
# : stl better; still, so far not really useful -- lots of structure left
plot1 <- ggplot(preprocessed_sales, aes(y = log_seasadj_scaled_sales.classical, x = date)) +
  geom_line()
plot2 <- ggplot(preprocessed_sales, aes(y = log_seasadj_scaled_sales.stl, x = date)) +
  geom_line()
grid.arrange(plot1, plot2, nrow=2)

# Not even a distant sign of stationarity
preprocessed_sales |>
  ACF(log_seasadj_scaled_sales.stl, lag_max=200) |>
  autoplot()

# Next: try a pipeline with differencings?
# 1. proceed with logged data
diffed_log_sales <- preprocessed_sales |>
  transmute(
    diff_1 = difference(log_seasadj_scaled_sales.stl),
    diff_2 = difference(log_seasadj_scaled_sales.stl, differences=2),
    diff_lag_year = difference(log_seasadj_scaled_sales.stl, lag=12*30),
  ) 

plot1 <- ggplot(diffed_log_sales, aes(y = diff_1, x = date)) +
  geom_line()
plot2 <- ggplot(diffed_log_sales, aes(y = diff_2, x = date)) +
  geom_line()
plot3 <- ggplot(diffed_log_sales, aes(y = diff_lag_year, x = date)) +
  geom_line()
grid.arrange(plot1, plot2, plot3, nrow=3)

diffed_log_sales |>
  ACF(diff_1, lag_max=200) |>
  autoplot()
# Conclusion: this is definetely better, but still far from stationary:
# To say the least, the variance is not constant across the series.
# One noteable feature persists: we've definitely left some seasonal structure in the data --
# variance at the start/end of each year is different.

# 2. regular (no log) differences
diffed_sales <- sales_grouped |>
  mutate(
    total_sales = impute_outlier(total_sales)
  ) |>
  model(
    stl = STL(total_sales ~ trend() +
          season(),
        robust = TRUE),
  ) |>
  components() |>
  select(season_adjust) |>
  mutate(
    seasadj_scaled_sales = min_max_scale(season_adjust)
  ) |>
  transmute(
    diff_1 = difference(seasadj_scaled_sales),
    diff_2 = difference(seasadj_scaled_sales, differences=2),
    diff_lag_year = difference(seasadj_scaled_sales, lag=12*30),
  ) 
plot1 <- ggplot(diffed_sales, aes(y = diff_1, x = date)) +
  geom_line()
plot2 <- ggplot(diffed_sales, aes(y = diff_2, x = date)) +
  geom_line()
plot3 <- ggplot(diffed_sales, aes(y = diff_lag_year, x = date)) +
  geom_line()
grid.arrange(plot1, plot2, plot3, nrow=3)

plot1 <- diffed_log_sales |>
  ACF(diff_1, lag_max=20) |>
  autoplot() +
  ylab("diffed_log_sales acf")
plot2 <- diffed_sales |>
  ACF(diff_1, lag_max=20) |>
  autoplot() + 
  ylab("diffed_sales acf")
grid.arrange(plot1, plot2, nrow=2)
# Conclusion: seem to be comparable;
# In this case I'd choose unlogged for the sake of simplicity and interpretability.
#
# One interesting observation from the ACF plot -- lag-1, lag-6 and 7 were found to be
# significant.
# Now, it is not necessarily interpretable, since the series is not stationary, but it is
# definitely interesting.

diffed_sales <- diffed_sales |>
  select(diff_1) |>
  drop_na()

# One last note: for technical reason I will be working with undifferenced data.
# We expect ARIMA to do differencing for us.
train <-  sales_grouped |>
  mutate(
    total_sales = impute_outlier(total_sales)
  ) |>
  model(
    stl = STL(total_sales ~ trend() +
                season(),
              robust = TRUE),
  ) |>
  components() |>
  select(season_adjust) |>
  mutate(
    seasadj_scaled_sales = min_max_scale(season_adjust)
  )
plot.ts(train$season_adjust)

##########

#################
# 2. model total sales
#################
total_sales_fits <- train |>
  model(
    Mean = MEAN(season_adjust),
    Naive = NAIVE(season_adjust),
    # Drift = DRIFT,
    ets = ETS(season_adjust ~ error() + trend() + season()),
    # note that we decide to model seasonality as well
    arima = ARIMA(season_adjust ~ pdq()),
    nonseasonal_arima = ARIMA(season_adjust ~ pdq() + PDQ(0,0,0))
  )

total_sales_fits2 <- sales_grouped |>
  mutate(
    total_sales = impute_outlier(total_sales),
  ) |>
  model(
    stl_arima = decomposition_model(
      STL(total_sales ~ trend() + season(), robust = TRUE),
      ARIMA(season_adjust ~ pdq())
    ),
    stl_arima_nonseasonal = decomposition_model(
      STL(total_sales ~ trend() + season(), robust = TRUE),
      ARIMA(season_adjust ~ pdq() + PDQ(0,0,0))
    ),
    log_stl_arima = decomposition_model(
      STL(log(total_sales) ~ trend() + season(), robust = TRUE),
      ARIMA(season_adjust ~ pdq())
    ),
    stl_ets = decomposition_model(
      STL(total_sales ~ trend() + season(), robust = TRUE),
      ETS(season_adjust ~ error() + trend() + season())
    ),
    stl_mean = decomposition_model(
      STL(total_sales ~ trend() + season(), robust = TRUE),
      MEAN(season_adjust)
    ),
    stl_naive= decomposition_model(
      STL(total_sales ~ trend() + season(), robust = TRUE),
      NAIVE(season_adjust)
    )
  )
# he actually has chosen the moving average model.
# which also means that starting at some point the predictions will no longer be meaningful
total_sales_fits$arima
#
total_sales_fits$nonseasonal_arima

# Closer look at fit
augment(total_sales_fits2) |>
  filter(year(date) == 2016 & month(date) == 1) |>
  pivot_wider(
    id_cols = date, 
    names_from = .model, 
    values_from = .fitted,
    unused_fn = function(l) l[1] 
  ) |>
  ggplot(aes(x = date)) + 
  geom_line(aes(y = total_sales), color="grey") + 
  geom_line(aes(y = stl_arima), color="orange") + 
  geom_line(aes(y = log_stl_arima), color="pink")

# Best one: stl_arima
total_sales_fits2 |> accuracy()

# examine forecasts
total_sales_fits2 |>
  forecast(h=30) |>
  filter(.model == "stl_arima") |>
  autoplot(
    total_sales_fits2 |> 
      augment() |> 
      filter(.model == "stl_arima") |> 
      filter(year(date)>=2017) |> select(-.model),
    point_forecast = lst(mean, median),
  )

#########
# 2.0 TODO: use external data.
# 1. We have to make the external data stationary.
#     Avoid "spurious regression".
# 2. fit arima with those ts's as exogenous regressors
#########
oil <- readr::read_csv('D:/contest/kaggle/sales/oil.csv') |>
  as_tsibble(index = date) |>
  drop_na() |>
  rename(price = dcoilwtico)
holidays <- readr::read_csv('D:/contest/kaggle/sales/holidays_events.csv') |>
  as_tsibble(index = date, key = c(description))

# make oil stationary
oil |>
  ggplot() +
  geom_line(aes(date, price))
  
oil <- oil |>
  mutate(
    log_price_diff = difference(log(price)),
    price_diff = difference(price)
  ) |>
  drop_na()

p1 <- oil |>
  ggplot() +
  geom_line(aes(x=date, y=price_diff)) 
p2 <- oil |>
  ggplot() +
  geom_line(aes(x=date, y=log_price_diff)) 
grid.arrange(p1,p2, nrow=2)

total_sales_regression_fits <- sales_grouped |>
  left_join(oil, by="date") |>
  # TODO: another option -- drop_na
  replace_na(list(
    price = median(oil$price, na.rm = TRUE),
    log_price_diff = median(oil$log_price_diff, na.rm = TRUE),
    price_diff = median(oil$price_diff, na.rm = TRUE)
  )) |>
  left_join(holidays, by="date", multiple="any") |>
  replace_na(list(locale = "none", type="none")) |>
  select(-description, -transferred, -locale_name) |>
  model(
    arima_regr_price = decomposition_model(
      STL(total_sales ~ trend() + season(), robust = TRUE),
      ARIMA(season_adjust ~ 1 + type + locale + price_diff + pdq())
    ),
    arima_regr_log_price = decomposition_model(
      STL(total_sales ~ trend() + season(), robust = TRUE),
      ARIMA(season_adjust ~  1 + type + locale + log_price_diff + pdq())
    )
  )

# an improvement 88k -> 86k
total_sales_regression_fits |> accuracy()
#########

##############################
# 2.1
# How can we further improve on the total_sales prediciton?
# Well, we could model each shop, or each product category (family) separately.
# The hypothesis is that seasonal trends are different form category to category, or from shop to shop
# TODO.
# Problem: you then have to refit proportions.
##############################
sales_grouped_by_store <- sales |>
  group_by(store_nbr) |>
  index_by(date) |>
  summarise(total_sales=sum(sales))
sales_grouped_by_store <- sales_grouped_by_store
  replace_na(list(
    total_sales=median(sales_grouped_by_store$total_sales, na.rm = TRUE)
  ))
  
sales_grouped_by_family <- sales |>
  group_by(family) |>
  index_by(date) |>
  summarise(total_sales=sum(sales))
sales_grouped_by_family <- sales_grouped_by_family
  replace_na(list(
    total_sales=median(sales_grouped_by_family$total_sales, na.rm = TRUE)
  ))

sales_grouped_by_store_fit <- sales_grouped_by_store |>
  left_join(oil, by="date", multiple = "all") |>
  replace_na(list(
    price = median(oil$price, na.rm = TRUE),
    log_price_diff = median(oil$log_price_diff, na.rm = TRUE),
    price_diff = median(oil$price_diff, na.rm = TRUE)
  )) |>
  left_join(holidays, by="date", multiple="any") |>
  replace_na(list(locale = "none", type="none")) |>
  select(-description, -transferred, -locale_name) |>
  model(
    arima_regr_price = decomposition_model(
      STL(total_sales ~ trend() + season(), robust = TRUE),
      ARIMA(season_adjust ~ 1 + type + locale + log_price_diff + pdq())
    )
  )

sales_grouped_by_family_fit <- sales_grouped_by_family |>
  left_join(oil, by="date", multiple = "all") |>
  replace_na(list(
    price = median(oil$price, na.rm = TRUE),
    log_price_diff = median(oil$log_price_diff, na.rm = TRUE),
    price_diff = median(oil$price_diff, na.rm = TRUE)
  )) |>
  left_join(holidays, by="date", multiple="any") |>
  replace_na(list(locale = "none", type="none")) |>
  select(-description, -transferred, -locale_name) |>
  model(
    arima_regr_price = decomposition_model(
      STL(total_sales ~ trend() + season(), robust = TRUE),
      ARIMA(season_adjust ~ 1 + type + locale + log_price_diff + pdq())
    )
  )

sales_grouped_by_family_fit |> accuracy()
sales_grouped_by_family_fit |> 
  augment() |>
  drop_na()
#################
# 3. model the proportion of sales for each product category for each shop
# Proportion = (store_nbr, family) / total sales THIS day
#################
sales_proportions <- sales |>
  # total
  index_by(date) |>
  mutate(
    total = sum(sales), 
    total_proportion = sales / total
  ) |>
  ungroup() |>
  # in_store
  group_by(store_nbr) |>
  index_by(date) |>
  mutate(
    in_store_total = sum(sales),
    in_store_proportion = sales / in_store_total
  ) |>
  ungroup() |>
  # in_store by family
  # TODO: does not work
  group_by(family, store_nbr) |>
  index_by(date) |>
  print() |>
  mutate(
    family_agg_in_store_proportion = sum(sales) / in_store_total
  ) |>
  ungroup() |>
  as_tsibble(index = date, key = c(store_nbr, family)) |>
  replace_na(list(total_proportion = 0, in_store_proportion = 0, family_agg_in_store_proportion = 0))

sales_proportions
# sanity_check
# For each date, for each store, sum of in_store_proportions adds up to 1.
# For each date, sum of total_proportions adds up to 1.
sales_proportions |>
  filter(date == "2016-01-02") |>
  group_by(store_nbr) |>
  summarise(
    agg_total_proportion = sum(total_proportion), 
    agg_in_store_proportion = sum(in_store_proportion)
  ) |>
  print() |>
  summarise(s = sum(agg_total_proportion))
# Our goal is to model total_proportion.
# We'll use in_store_proportions (which is simply a scaled version of the former) for EDA.
# We'll show total_proportion of sales for those categories, which have significant average in_store_proportion.
popular_first_store <- sales_proportions |>
  filter(store_nbr == 1) |>
  group_by(family) |>
  mutate(
    by_family = mean(in_store_proportion)
  ) |>
  ungroup() |>
  filter(by_family > 0.05)

popular_first_store |>
  ggplot(aes(y=total_proportion, x=date, color=family)) +
  geom_line()

# Will use only >= 2016 for prediction for this reason.
# Reasons:
# 1. before ~mid 2015 the sales, for some reason, exhibit strange patterns
# 2. Computational
# There are anomalies in this data as well, sudden bursts and declines, including
# those at the start/end of each year.
# Apart form that, we see that the proportions are more or less stationary.
p1 <- popular_first_store |>
  filter(year(date) >= 2016) |>
  ggplot(aes(y=total_proportion, x=date, color=family)) +
  geom_line()
p2 <- popular_first_store |>
  filter(year(date) >= 2016) |>
  ggplot(aes(y=in_store_proportion, x=date, color=family)) +
  geom_line()
grid.arrange(p1,p2,nrow=2)

# Modeling.
sales_proportions <- sales_proportions |>
  filter(year(date) >= 2016)

# For each (store_nbr, family) [which are our keys], we get a model.
# Takes a while, for there are 1_782 such pairs
# Takes about ten minutes.
sales_proportions
plan(multisession)
proportion_fits <- sales_proportions |>
  model(
    mean = MEAN(sqrt(total_proportion)),
    sqrt_arima = ARIMA(sqrt(total_proportion) ~ pdq()),
    log_arima = ARIMA(log(total_proportion) ~ pdq()),
    ets = ETS(sqrt(total_proportion)),
    .safely = TRUE,
  )
plan(sequential)

metrics <- proportion_fits |>
  accuracy(measures = lst(RMSE)) |>
  pivot_wider(names_from = .model, values_from = RMSE) |>
  select(-.type)

proportion_best_fits <- proportion_fits |>
  transmute(
    best_fit = if_else(
      metrics$sqrt_arima < metrics$ets & 
      metrics$sqrt_arima < metrics$mean,
      sqrt_arima,
      if_else(
        metrics$ets < metrics$mean,
        ets, mean
      )
    )
  )

proportion_fcs <- proportion_best_fits |>
  forecast(h=12)

first_store_fit <- proportion_best_fits |>
  filter(store_nbr == 1) |>
  augment()

# examine fit
popular_first_store |>
  filter(year(date) == 2016 & month(date) == 5) |>
  left_join(first_store_fit, by=c("date", "store_nbr", "family")) |>
  print() |>
  ggplot() +
  geom_line(aes(date, .fitted), color="pink") +
  geom_line(aes(date, total_proportion.y))

# There is a slight problem: we in no way ensure fitted proportions to add up to one
# Fortunately, it seems that it is satisfied:
sales_proportions |>
  filter(year(date) == 2016 & month(date) == 5) |>
  left_join(augment(proportion_best_fits), by=c("date", "store_nbr", "family")) |>
  index_by(date) |>
  summarise(t.fitted = sum(.fitted), t = sum(total_proportion.x))

# metrics
metrics

#################

###################################
# 3.0 It may be helpful to include exogenous holdiay info to predict proportions
# TODO.
###################################

#####################################
# 4. Combine into prediction
#####################################
h <- test |>
  count(day(date)) |>
  nrow()

proportion_fcs <- proportion_best_fits |>
  forecast(h=h)

proportion_fcs2 <- readr::read_csv("D:/contest/kaggle/sales/percent_fcs.csv") |>
  as_tsibble(index = date, key = c(store_nbr, family))

total_sales_fcs <- total_sales_fits2 |>
  forecast(h=h) |>
  filter(.model == "stl_arima") |>
  select(-.model)

total_sales_regr_fcs <- total_sales_regression_fits |>
  forecast(
    new_data = test |>
      as_tibble() |>
      group_by(date) |>
      summarise() |>
      as_tsibble(index=date) |>
      left_join(oil, by="date", multiple = "all") |>
      replace_na(list(
        price_diff = median(oil$price_diff, na.rm = TRUE),
        log_price_diff = median(oil$log_price_diff, na.rm = TRUE)
      )) |>
      left_join(holidays, by="date", multiple="any") |>
      replace_na(list(locale = "none", type="none")) |>
      select(-description, -transferred, -locale_name)
  ) |>
  filter(.model == "arima_regr_price") |>
  select(-.model)

prediction <- test |>
  as_tsibble(index = date, key = c(store_nbr, family)) |>
  left_join(proportion_fcs, by=c("date", "store_nbr", "family")) |>
  select(-total_proportion) |>
  rename(total_proportion = .mean) |>
  as_tibble() |>
  right_join(total_sales_fcs, by=c("date"), multiple="all") |>
  mutate(sales = total_proportion * .mean)

prediction2 <- test |>
  as_tsibble(index = date, key = c(store_nbr, family)) |>
  left_join(proportion_fcs2, by=c("date", "store_nbr", "family")) |>
  rename(total_proportion = percent_fc) |>
  as_tibble() |>
  right_join(total_sales_fcs, by=c("date"), multiple="all") |>
  mutate(sales = total_proportion * .mean)

readr::write_csv(prediction |> select(id, sales), "D:/contest/kaggle/sales/submission_02.csv")

# visual
merge(
  x = prediction |>
    as_tsibble(index = date, key = c(store_nbr, family)) |>
    filter(year(date)>=2017 & store_nbr == 1 & family == "AUTOMOTIVE") |> 
    select(-.mean, -id, -total_sales, -total_proportion, -store_nbr, -family) |>
    as_tibble(),
  y = sales |> 
    filter(year(date)>=2017 & store_nbr == 1 & family == "AUTOMOTIVE") |> 
    select(-onpromotion, -id, -store_nbr, -family) |>
    as_tibble(),
  all = T
)

prediction |>
  as_tsibble(index = date, key = c(store_nbr, family)) |>
  filter(year(date)>=2017 & store_nbr == 1 & family == "AUTOMOTIVE") |> 
  select(-.mean, -id, -total_sales, -total_proportion, -store_nbr, -family) |>
  ggplot() + geom_line(aes(date, sales))

sales |> 
  filter(year(date)>=2017 & store_nbr == 1 & family == "AUTOMOTIVE") |> 
  select(-onpromotion, -id, -store_nbr, -family) |>
  ggplot() + geom_line(aes(date, sales))

ggplot() +
geom_line(
  sales |> 
    filter(year(date)>=2017 & store_nbr == 1 & family == "AUTOMOTIVE") |> 
    select(-onpromotion, -id),
  aes(date, sales)
) +
geom_line(
  prediction |>
    as_tsibble(index = date, key = c(store_nbr, family)) |>
    filter(year(date)>=2017 & store_nbr == 1 & family == "AUTOMOTIVE") |> 
    select(-.mean, -id, -total_sales, -total_proportion),
  aes(date, sales)
)

prediction

proportion_fcs

#####################################


#####
#decomposition
#####
sales_dcmp <- sales_grouped |>
  model(
    STL(total_sales ~ trend() +
          season(),
        robust = TRUE)
  ) |>
  components()
  

sales_dcmp |> autoplot()

sales_dcmp |>
  select(total_sales, season_adjust) |>
  autoplot()

sales_adj_train <- sales_dcmp |>
  select(date, season_adjust, total_sales) |>
  filter(year(date) <= 2016)

sales_adj_val <- sales_dcmp |>
  select(date, season_adjust, total_sales) |>
  filter(year(date) > 2016)

sales_adj_fits <- sales_adj_train |>
  model(
    Mean = MEAN(season_adjust),
    Naive = NAIVE(season_adjust),
    # Drift = DRIFT,
    ets = ETS(season_adjust ~ error("A") + trend() + season("N")),
    arima = ARIMA(season_adjust ~ pdq() + PDQ(0, 0, 0))
  )
sales_adj_fits_fcs <- sales_adj_fits |>
  forecast(h = nrow(sales_adj_val))

(sales_adj_fits$arima)

sales_adj_fits_fcs |>
  autoplot(sales_adj_train, level = NULL) +
  autolayer(
    sales_adj_val,
    colour = "black"
  )

sales_adj_log <- sales_adj_train |>
  mutate(log_sales = log(season_adjust), .keep="all")
autoplot(sales_adj_log, .vars = log_sales)
  
?ETS
#
train <- sales_grouped |>
  filter(year(date) <= 2016)

val <- sales_grouped |>
  filter(year(date) > 2016)

fits <- train |> 
  model(
    stlf = decomposition_model(
      STL(total_sales ~ trend(window = 7), robust = TRUE),
      NAIVE(season_adjust)
    ),
    Mean = MEAN(total_sales),
    Naive = NAIVE(total_sales),
    SNaive = SNAIVE(total_sales)
  )
decomposition_model
dim(val)[1]
nrow(val)

fits_fcs <- fits |>
  forecast(h = nrow(val))

?filter_index

fits_fcs |>
  autoplot(train, level = NULL) +
  autolayer(
    val,
    colour = "black"
  )

accuracy(fits_fcs, val)
#####

