#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <memory>

namespace tcm {
namespace recommend {

struct AssociationRule {
    std::vector<std::string> antecedent;
    std::vector<std::string> consequent;
    double support;
    double confidence;
    double lift;
    double avg_efficacy;
};

struct AcupointCombination {
    std::vector<std::string> acupoints;
    double support;
    double avg_pain_relief;
    double avg_deqi;
    int sample_count;
};

struct Transaction {
    std::unordered_set<std::string> acupoints_set;
    double pain_relief;
    double deqi_intensity;
};

struct Itemset {
    std::set<std::string> items;
    int count;
    double total_pain_relief;
    double total_deqi;
};

class AcupunctureRecommender {
public:
    AcupunctureRecommender();
    ~AcupunctureRecommender();

    void add_transaction(const std::vector<std::string>& acupoints,
                         double pain_relief,
                         double deqi_intensity);

    void run_apriori(double min_support = 0.05,
                     double min_confidence = 0.3,
                     double min_lift = 1.2,
                     int max_len = 5,
                     int num_threads = 0);

    std::vector<AcupointCombination> get_recommendations(
        const std::vector<std::string>& current_acupoints,
        int top_k = 10,
        double min_lift = 1.2) const;

    std::vector<AssociationRule> get_rules(int top_k = 20, double min_lift = 0.0) const;
    std::vector<AcupointCombination> get_frequent_itemsets(int min_count = 5) const;

    size_t transaction_count() const { return transactions_.size(); }
    size_t rule_count() const { return rules_.size(); }
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::vector<Transaction> transactions_;
    std::vector<AssociationRule> rules_;
    std::vector<Itemset> frequent_itemsets_;
    double min_support_;
    double min_confidence_;
    double min_lift_;
    int num_threads_;

    std::vector<Itemset> generate_candidates(const std::vector<Itemset>& prev, int k) const;
    int count_support(const std::set<std::string>& itemset) const;
    double calc_avg_pain_relief(const std::set<std::string>& itemset) const;
    double calc_avg_deqi(const std::set<std::string>& itemset) const;

    std::vector<Itemset> parallel_count_supports(const std::vector<Itemset>& candidates,
                                                  int num_threads) const;

    void generate_rules_from_itemset(const Itemset& itemset);
    std::vector<std::vector<std::string>> get_subsets(const std::set<std::string>& s) const;

    static std::string itemset_key(const std::set<std::string>& s);
    static bool has_infix(const std::set<std::string>& set, const std::set<std::string>& subset);
};

} // namespace recommend
} // namespace tcm
