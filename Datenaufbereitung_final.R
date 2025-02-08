# _______________________________________________
#
# Packages laden
# _______________________________________________

library(tidyverse)
library(psych)
library(caret)
library(dplyr)
library(tidyr)
library(glmnet)
library(corrplot)
library(vip)
library(ggplot2)
library(rpart)
library(rpart.plot)


# _______________________________________________
#
# Datensatz erstellen
# _______________________________________________

# Datensatz einlesen und leere Zellen mit NAs füllen
data_raw <- read.csv("rpp_data.csv")
data_raw[data_raw == ""] <- NA

# Datensatz erstellen ohne Variablen, die die Replikationsstudien betreffen(erkennbar an "..R")
data_pre <- data_raw[, !grepl("..R.", names(data_raw))]

# Hinzufügen und Transformieren vom Kriterium (replicate) 
data_pre$replicate <- data_raw$Replicate..R.
data_pre$replicate <- ifelse(data_pre$replicate == "yes", "yes","no")
data_pre$replicate <- as.factor(data_pre$replicate)

# Subset ohne die NAs im Kriterium
data_pre <- subset(data_pre, !is.na(replicate))
data_pre <- data_pre %>%
  rename(Study_num = Study.Num) # Umbenennen der Variable Studiennummer, damit data_pre & data_scite zusammengefuehrt werden koennen

# Scite-Variablen hinzufuegen
data_scite <- read.csv("data_scite.csv")
data_pre <- merge(data_pre, data_scite, by = "Study_num")

# finalen Datensatz erstellen
data_final <- data.frame(study_title = data_pre$Study.Title..O.,     
                         replicate = data_pre$replicate)
table(data_pre$replicate) # Verteilung im Kriterium (61 nicht repliziert, 39 repliziert)


# _______________________________________________
#
# Prädiktorenblock 1
# _______________________________________________

# number_of_authors (3 Kategorien)
data_final$number_of_authors <- 3 
data_final$number_of_authors[data_pre$Number.of.Authors..O. == "1"] <- 1
data_final$number_of_authors[data_pre$Number.of.Authors..O. == "2"] <- 2
data_final$number_of_authors[data_pre$Number.of.Authors..O. == "3"] <- 2
data_final$number_of_authors <- as.factor(data_final$number_of_authors)

# citations_first_author
data_final$citations_first_author <- as.numeric(data_pre$Citation.Count..1st.author..O.)
data_final$citations_first_author[data_final$citations_first_author > 18734] <- NA # Ausreisser wegen grossen Varianz raus

# suprising_results
data_final$suprising_result <- as.numeric(data_pre$Surprising.result..O.)

# reported_p_value (3 Kategorien)
data_final$reported_p_value <- data_pre$Reported.P.value..O.

# Funktion zur Umwandlung der prep-Werte in p-Werte
prep_to_p <- function(prep_values) {
  # Berechnung von z-Werten aus den Prep-Werten
  z_values <- -sqrt(2) * qnorm(1 - prep_values)
  # Umwandlung der z-Werte in p-Werte
  p_values <- 2 * (1 - pnorm(abs(z_values)))
  return(p_values)
}

# Vektor der Prep-Werte
prep_values <- c(0.18, 0.92, 0.927, 0.93, 0.947, 0.95, 0.97, 0.99)
# Umwandlung in p-Werte
p_values <- prep_to_p(prep_values)
# Ausgabe der Ergebnisse
data.frame(PREP = prep_values, p_value = p_values)

# Werte im Datensatz umaendern
data_final$reported_p_value[data_final$reported_p_value == "prep = .18"] <- "0.195485271"
data_final$reported_p_value[data_final$reported_p_value == "prep = .92"] <- "0.046914501"
data_final$reported_p_value[data_final$reported_p_value == "prep = .927"] <- "0.039783223"
data_final$reported_p_value[data_final$reported_p_value == "prep = .93"] <- "0.036880570"
data_final$reported_p_value[data_final$reported_p_value == "prep = .947"] <- "0.022255064"
data_final$reported_p_value[data_final$reported_p_value == "prep = .95"] <- "0.020009254"
data_final$reported_p_value[data_final$reported_p_value == "prep = .97"] <- "0.007817689"
data_final$reported_p_value[data_final$reported_p_value == "prep = .99"] <- "0.001002042"

# "X" auf NA setzen
data_final$reported_p_value[data_final$reported_p_value == "X"] <- NA

# Erstellen von 3 Kategorien
data_final$reported_p_value[data_final$reported_p_value == "<.00001"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "<.0001"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "<.001"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "<.005"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "<.007"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "<.01"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "<0.001"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "<0.005"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "0"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "0.005"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "0.007817689"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "0.008"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "0.01"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "prep > .99"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "0.00003"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "0.001"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "0.001002042"] <- "<.01"
data_final$reported_p_value[data_final$reported_p_value == "0.002"] <- "<.01"

data_final$reported_p_value[data_final$reported_p_value == "<.02"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "<.05"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "<0.05"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == ">0.1"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.02"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.020009254"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.022255064 "] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.023"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.028"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.03"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.036880570 "] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.037"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.039783223"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.046914501"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.05"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == '"significant"'] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.022255064"] <- "<.05"
data_final$reported_p_value[data_final$reported_p_value == "0.036880570"] <- "<.05"

data_final$reported_p_value[data_final$reported_p_value == "<.06"] <- ">.05"
data_final$reported_p_value[data_final$reported_p_value == "0.195485271"] <- ">.05"
data_final$reported_p_value[data_final$reported_p_value == "0.48"] <- ">.05"

table(data_final$reported_p_value)
data_final$reported_p_value <- as.factor(data_final$reported_p_value)

# type_of_effect (3 Kategorien)
data_final$type_of_effect <- 3 # 3: sonstiges
data_final$type_of_effect[data_pre$Type.of.effect..O. == "main effect"] <- 1
data_final$type_of_effect[data_pre$Type.of.effect..O. == "interaction"] <- 2
data_final$type_of_effect <- as.factor(data_final$type_of_effect)

# min_power_quotient
data_pre$X80..power <- as.numeric(data_pre$X80..power) # notwendiges N bei einer Power von .80
data_pre$N..O. <- as.numeric(data_pre$N..O.) # N in Originalstudie 
data_final$min_power_quotient <- data_pre$N..O./ data_pre$X80..power 

# effsize (umgewandelt in r)
data_final$effsize <- data_pre$T_r..O.

# conceptual_replications
data_final$conceptual_replications <- data_pre$Internal.conceptual.replications..O.

# Praediktorenblock 1
Präd_b1 <- names(data_final[!(names(data_final) %in% c("replicate", "study_title"))])


# _______________________________________________
#
# Prädiktorenblock 2
# _______________________________________________

# methodology_expertise_required (5 Kategorien)
data_final$methodology_expertise_required <- factor(data_pre$Methodology.expertise.required..O., 
                                                    levels = c("No expertise required", "Slight expertise required", "Moderate expertise required", "Strong expertise required", "Extreme expertise required"))
data_final$methodology_expertise_required <- as.numeric(data_final$methodology_expertise_required)

# conceptual_replications_success
data_final$conceptual_replications_success <- data_pre$Successful.conceptual.replications..O./data_pre$Internal.conceptual.replications..O. # Anzahl erfolgreiche Replikationen / Anzahl durchgefuehrter Repliaktionen
data_final$conceptual_replications_success[is.na(data_final$conceptual_replications_success) | 
                                             is.infinite(data_final$conceptual_replications_success)] <- 0
# n
data_final$n <- data_pre$N..O.
data_final$n[data_final$n > 10000] <- NA # Ausreißer raus

# disciplin (2 Kategorien)
data_final$discipline <- as.factor(data_pre$Discipline..O.) # Cognitive = 1, Social = 2

# journal (3 Kategorien)
data_final$journal <- factor(data_pre$Journal..O., levels = c("JEPLMC", "JPSP", "PS"))

# citation_count
data_final$citation_count <- as.numeric(data_pre$Citation.count..paper..O.)
data_final$citation_count[data_final$citation_count > 298] <- NA # Ausreisser raus wegen grossen Varianz

# Praediktorenblock 2
Präd_b2 <- names(data_final[!(names(data_final) %in% c("replicate", "study_title", Präd_b1))])


# _______________________________________________
#
# Prädiktorenblock 3
# _______________________________________________

# institution_prestige_first_author
data_final$institution_prestige_first_author <- data_pre$Institution.prestige..1st.author..O.

# opportunity_for_lod (lack of diligence) (5 Kategorien)
data_final$opportunity_for_lod <- factor(data_pre$Opportunity.for.lack.of.diligence..O.,
                                         levels = c("No opportunity for lack of diligence to affect the results", "Slight opportunity for lack of diligence to affect the results",
                                                    "Moderate opportunity for lack of diligence to affect the results", "Strong opportunity for lack of diligence to affect the results",
                                                    "Extreme opportunity for lack of diligence to affect the results"))
data_final$opportunity_for_lod <- as.numeric(data_final$opportunity_for_lod)

# opportunity_for_expectancy (5 Kategorien)
data_final$opportunity_for_expectancy <- factor(data_pre$Opportunity.for.expectancy.bias..O., 
                                               levels = c("No opportunity for researcher expectations to influence results", "Slight opportunity for researcher expectations to influence results", 
                                                          "Moderate opportunity for researcher expectations to influence results", "Strong opportunity for researcher expectations to influence results",
                                                          "Extreme opportunity for researcher expectations to influence results"))
data_final$opportunity_for_expectancy <- as.numeric(data_final$opportunity_for_expectancy)

# importance_effect
data_final$importance_effect <- as.numeric(data_pre$Exciting.result..O.)

# material_collected (2 Kategorien)
data_final$material_collected <- 0
data_final$material_collected[data_pre$Collect.materials.from.authors == "Complete"] <- 1
data_final$material_collected <- as.factor(data_final$material_collected)

# number_of_studies
data_final$number_of_studies <- as.numeric(data_pre$Number.of.Studies..O.)

# Praediktorenblock 3
Präd_b3 <- names(data_final[!(names(data_final) %in% c("replicate", "study_title", Präd_b1, Präd_b2))])

# Praediktorenblock mit allen Praediktoren
Präd_b1_b2_b3 <- names(data_final[!(names(data_final) %in% c("replicate", "study_title"))])


# _______________________________________________
#
# data_final finalisieren
# _______________________________________________

data_final$study_title <- NULL # Titel entfernen
data_corr <- data_final


# _______________________________________________
#
# Modelle spezifizieren
# _______________________________________________

# Variablennamen der Praediktorenbloecke mit dem Trennzeichen "+" verbinden
# fuer jeden Praediktorenblock ein Modell erstellen
mod_b1 <- as.formula(paste("replicate ~" , paste(Präd_b1, collapse =" + ")))   
mod_b2 <- as.formula(paste("replicate ~" , paste(Präd_b2, collapse =" + ")))   
mod_b3 <- as.formula(paste("replicate ~" , paste(Präd_b3, collapse =" + ")))   
mod_b1_b2_b3 <- as.formula(paste("replicate ~" , paste(Präd_b1_b2_b3, collapse =" + ")))


# # _______________________________________________
# #
# # scite-Variablen  
# # _______________________________________________
# 
# # Scite_citing_pub
# data_corr$scite_total <- as.numeric(data_pre$scite_citing_pub)
# 
# # Scite_mentioning
# data_corr$scite_mention <- as.numeric(data_pre$scite_mentioning)
# 
# # scite_supporting
# data_corr$scite_support <- as.numeric(data_pre$scite_supporting)
# 
# # scite_contradicting
# data_corr$scite_contra <- as.numeric(data_pre$scite_contradicting)


# _______________________________________________
#
# Korrelationen  
# _______________________________________________

data_corr <- data_corr %>% mutate_if(is.factor, ~ as.numeric(.))
corr_matrix <- cor(data_corr, use = "complete.obs")
print(corr_matrix)
corrplot(corr_matrix)
corrplot(corr_matrix, method = "circle")





