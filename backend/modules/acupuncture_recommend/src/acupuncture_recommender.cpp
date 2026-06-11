#include "acupuncture_recommender.h"
#include "../../common/include/thread_pool.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <thread>

namespace tcm {
namespace recommend {

struct AcupunctureRecommender::Impl {
    common::ThreadPool pool;
};

AcupunctureRecommender::AcupunctureRecommender()
    : min_support_(0.05), min_confidence_(0.3), min_lift_(1.2), num_threads_(0) {
}

AcupunctureRecommender::~AcupunctureRecommender() = default;

void AcupunctureRecommender::add_transaction(
    const std::vector<std::string>& acupoints,
    double pain_relief,
    double deqi_intensity) {
    Transaction t;
    for (const auto& ap : acupoints) {
        t.acupoints_set.insert(ap);
    }
    t.pain_relief = pain_relief;
    t.deqi_intensity = deqi_intensity;
    transactions_.push_back(t);
}

void AcupunctureRecommender::clear() {
    transactions_.clear();
    rules_.clear();
    frequent_itemsets_.clear();
}

std::vector<AcupunctureRecommender::Itemset>
AcupunctureRecommender::parallel_count_supports(
    const std::vector<Itemset>& candidates,
    int num_threads) const {

    if (candidates.size() < 4 || num_threads <= 1) {
        std::vector<Itemset> result = candidates;
        for (auto& cand : result) {
            cand.count = count_support(cand.items);
            if (cand.count > 0) {
                cand.total_pain_relief = calc_avg_pain_relief(cand.items) * cand.count;
                cand.total_deqi = calc_avg_deqi(cand.items) * cand.count;
            }
        }
        return result;
    }

    size_t n = candidates.size();
    size_t chunk_size = (n + num_threads - 1) / num_threads;

    std::vector<std::vector<Itemset>> chunks(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(n, start + chunk_size);
        if (start >= end) break;
        chunks[i].reserve(end - start);
        for (size_t j = start; j < end; ++j) {
            chunks[i].push_back(candidates[j]);
        }
    }

    std::vector<std::future<std::vector<Itemset>>> futures;
    for (int i = 0; i < num_threads; ++i) {
        if (chunks[i].empty()) continue;
        futures.push_back(std::async(std::launch::async,
            [this, &chunk = chunks[i]]() -> std::vector<Itemset> {
                for (auto& cand : chunk) {
                    cand.count = count_support(cand.items);
                    if (cand.count > 0) {
                        cand.total_pain_relief = calc_avg_pain_relief(cand.items) * cand.count;
                        cand.total_deqi = calc_avg_deqi(cand.items) * cand.count;
                    }
                }
                return chunk;
            }));
    }

    std::vector<Itemset> result;
    result.reserve(candidates.size());
    for (auto& f : futures) {
        auto chunk_result = f.get();
        result.insert(result.end(), chunk_result.begin(), chunk_result.end());
    }
    return result;
}

void AcupunctureRecommender::run_apriori(double min_support, double min_confidence,
                                          double min_lift, int max_len, int num_threads) {
    min_support_ = min_support;
    min_confidence_ = min_confidence;
    min_lift_ = min_lift;
    num_threads_ = num_threads > 0 ? num_threads
                : (int)std::max(1u, std::thread::hardware_concurrency());

    rules_.clear();
    frequent_itemsets_.clear();

    if (transactions_.empty()) return;

    int min_count = (int)ceil(min_support * transactions_.size());
    if (min_count < 1) min_count = 1;

    std::map<std::string, int> item_counts;
    std::map<std::string, double> item_pain_sum;
    std::map<std::string, double> item_deqi_sum;

    for (const auto& t : transactions_) {
        for (const auto& ap : t.acupoints_set) {
            item_counts[ap]++;
            item_pain_sum[ap] += t.pain_relief;
            item_deqi_sum[ap] += t.deqi_intensity;
        }
    }

    std::vector<Itemset> prev_itemsets;
    for (const auto& kv : item_counts) {
        if (kv.second >= min_count) {
            Itemset is;
            is.items.insert(kv.first);
            is.count = kv.second;
            is.total_pain_relief = item_pain_sum[kv.first];
            is.total_deqi = item_deqi_sum[kv.first];
            prev_itemsets.push_back(is);
            frequent_itemsets_.push_back(is);
        }
    }

    for (int k = 2; k <= max_len && !prev_itemsets.empty(); ++k) {
        auto candidates = generate_candidates(prev_itemsets, k);

        auto counted = parallel_count_supports(candidates, num_threads_);

        std::vector<Itemset> curr_itemsets;
        for (auto& cand : counted) {
            if (cand.count >= min_count) {
                curr_itemsets.push_back(cand);
                frequent_itemsets_.push_back(cand);
            }
        }

        prev_itemsets = curr_itemsets;
    }

    for (const auto& itemset : frequent_itemsets_) {
        if (itemset.items.size() >= 2) {
            generate_rules_from_itemset(itemset);
        }
    }

    std::sort(rules_.begin(), rules_.end(),
        [](const AssociationRule& a, const AssociationRule& b) {
            return a.confidence > b.confidence;
        });
}

std::vector<AcupunctureRecommender::Itemset>
AcupunctureRecommender::generate_candidates(const std::vector<Itemset>& prev, int k) const {
    std::vector<Itemset> candidates;

    for (size_t i = 0; i < prev.size(); ++i) {
        for (size_t j = i + 1; j < prev.size(); ++j) {
            std::set<std::string> merged = prev[i].items;
            merged.insert(prev[j].items.begin(), prev[j].items.end());

            if ((int)merged.size() != k) continue;

            bool exists = false;
            for (const auto& c : candidates) {
                if (c.items == merged) { exists = true; break; }
            }
            if (exists) continue;

            bool valid = true;
            if (k > 2) {
                std::vector<std::string> v(merged.begin(), merged.end());
                for (size_t m = 0; m < v.size(); ++m) {
                    std::set<std::string> subset;
                    for (size_t n = 0; n < v.size(); ++n) {
                        if (n != m) subset.insert(v[n]);
                    }
                    bool found = false;
                    for (const auto& p : prev) {
                        if (p.items == subset) { found = true; break; }
                    }
                    if (!found) { valid = false; break; }
                }
            }

            if (valid) {
                Itemset is;
                is.items = merged;
                is.count = 0;
                is.total_pain_relief = 0;
                is.total_deqi = 0;
                candidates.push_back(is);
            }
        }
    }

    return candidates;
}

int AcupunctureRecommender::count_support(const std::set<std::string>& itemset) const {
    int count = 0;
    for (const auto& t : transactions_) {
        bool all_found = true;
        for (const auto& item : itemset) {
            if (t.acupoints_set.find(item) == t.acupoints_set.end()) {
                all_found = false;
                break;
            }
        }
        if (all_found) count++;
    }
    return count;
}

double AcupunctureRecommender::calc_avg_pain_relief(const std::set<std::string>& itemset) const {
    double total = 0;
    int count = 0;
    for (const auto& t : transactions_) {
        bool all_found = true;
        for (const auto& item : itemset) {
            if (t.acupoints_set.find(item) == t.acupoints_set.end()) {
                all_found = false;
                break;
            }
        }
        if (all_found) {
            total += t.pain_relief;
            count++;
        }
    }
    return count > 0 ? total / count : 0.0;
}

double AcupunctureRecommender::calc_avg_deqi(const std::set<std::string>& itemset) const {
    double total = 0;
    int count = 0;
    for (const auto& t : transactions_) {
        bool all_found = true;
        for (const auto& item : itemset) {
            if (t.acupoints_set.find(item) == t.acupoints_set.end()) {
                all_found = false;
                break;
            }
        }
        if (all_found) {
            total += t.deqi_intensity;
            count++;
        }
    }
    return count > 0 ? total / count : 0.0;
}

void AcupunctureRecommender::generate_rules_from_itemset(const Itemset& itemset) {
    auto subsets = get_subsets(itemset.items);

    for (const auto& subset : subsets) {
        if (subset.empty()) continue;
        std::set<std::string> ant_set(subset.begin(), subset.end());
        if (ant_set == itemset.items) continue;

        std::set<std::string> cons_set;
        for (const auto& item : itemset.items) {
            if (ant_set.find(item) == ant_set.end()) {
                cons_set.insert(item);
            }
        }
        if (cons_set.empty()) continue;

        int ant_count = count_support(ant_set);
        if (ant_count == 0) continue;

        double support = (double)itemset.count / transactions_.size();
        double confidence = (double)itemset.count / ant_count;

        int cons_count = count_support(cons_set);
        double cons_support = (double)cons_count / transactions_.size();
        double lift = cons_support > 0 ? confidence / cons_support : 0.0;

        if (confidence >= min_confidence_ && lift >= min_lift_) {
            AssociationRule rule;
            rule.antecedent = std::vector<std::string>(ant_set.begin(), ant_set.end());
            rule.consequent = std::vector<std::string>(cons_set.begin(), cons_set.end());
            rule.support = support;
            rule.confidence = confidence;
            rule.lift = lift;
            rule.avg_efficacy = itemset.total_pain_relief / itemset.count;
            rules_.push_back(rule);
        }
    }
}

std::vector<std::vector<std::string>>
AcupunctureRecommender::get_subsets(const std::set<std::string>& s) const {
    std::vector<std::vector<std::string>> subsets;
    std::vector<std::string> v(s.begin(), s.end());
    int n = (int)v.size();

    for (int mask = 1; mask < (1 << n) - 1; ++mask) {
        std::vector<std::string> subset;
        for (int i = 0; i < n; ++i) {
            if (mask & (1 << i)) subset.push_back(v[i]);
        }
        if (!subset.empty()) subsets.push_back(subset);
    }

    return subsets;
}

std::vector<AcupointCombination>
AcupunctureRecommender::get_recommendations(
    const std::vector<std::string>& current_acupoints,
    int top_k,
    double min_lift) const {

    std::set<std::string> current_set(current_acupoints.begin(), current_acupoints.end());
    std::map<std::string, double> candidate_scores;
    std::map<std::string, double> candidate_pain_relief;
    std::map<std::string, double> candidate_deqi;
    std::map<std::string, int> candidate_count;

    for (const auto& rule : rules_) {
        if (rule.lift < min_lift) continue;

        std::set<std::string> ant_set(rule.antecedent.begin(), rule.antecedent.end());

        bool all_match = true;
        for (const auto& a : ant_set) {
            if (current_set.find(a) == current_set.end()) {
                all_match = false;
                break;
            }
        }
        if (!all_match) continue;

        for (const auto& c : rule.consequent) {
            if (current_set.find(c) != current_set.end()) continue;

            double score = rule.confidence * rule.lift * rule.avg_efficacy;
            if (candidate_scores.find(c) == candidate_scores.end() || score > candidate_scores[c]) {
                candidate_scores[c] = score;
                candidate_pain_relief[c] = rule.avg_efficacy;
            }
        }
    }

    for (const auto& is : frequent_itemsets_) {
        if (is.items.size() < 2) continue;

        bool contains_all_current = true;
        for (const auto& ca : current_set) {
            if (is.items.find(ca) == is.items.end()) {
                contains_all_current = false;
                break;
            }
        }
        if (!contains_all_current) continue;

        for (const auto& item : is.items) {
            if (current_set.find(item) != current_set.end()) continue;

            double avg_pr = is.total_pain_relief / std::max(1, is.count);
            double score = (double)is.count / transactions_.size() * avg_pr;

            if (candidate_scores.find(item) == candidate_scores.end() ||
                score > candidate_scores[item]) {
                candidate_scores[item] = score;
                candidate_pain_relief[item] = avg_pr;
                candidate_deqi[item] = is.total_deqi / std::max(1, is.count);
                candidate_count[item] = is.count;
            }
        }
    }

    std::vector<AcupointCombination> result;
    for (const auto& kv : candidate_scores) {
        AcupointCombination ac;
        ac.acupoints.push_back(kv.first);
        ac.support = 0;
        ac.avg_pain_relief = candidate_pain_relief[kv.first];
        ac.avg_deqi = candidate_deqi[kv.first];
        ac.sample_count = candidate_count[kv.first];
        result.push_back(ac);
    }

    std::sort(result.begin(), result.end(),
        [&candidate_scores](const AcupointCombination& a, const AcupointCombination& b) {
            return candidate_scores.at(a.acupoints[0]) > candidate_scores.at(b.acupoints[0]);
        });

    if ((int)result.size() > top_k) result.resize(top_k);
    return result;
}

std::vector<AssociationRule>
AcupunctureRecommender::get_rules(int top_k, double min_lift) const {
    std::vector<AssociationRule> result;
    for (const auto& r : rules_) {
        if (r.lift >= min_lift) result.push_back(r);
    }
    if ((int)result.size() > top_k) result.resize(top_k);
    return result;
}

std::vector<AcupointCombination>
AcupunctureRecommender::get_frequent_itemsets(int min_count) const {
    std::vector<AcupointCombination> result;
    for (const auto& is : frequent_itemsets_) {
        if (is.count >= min_count) {
            AcupointCombination ac;
            ac.acupoints = std::vector<std::string>(is.items.begin(), is.items.end());
            ac.support = transactions_.empty() ? 0 : (double)is.count / transactions_.size();
            ac.avg_pain_relief = is.count > 0 ? is.total_pain_relief / is.count : 0.0;
            ac.avg_deqi = is.count > 0 ? is.total_deqi / is.count : 0.0;
            ac.sample_count = is.count;
            result.push_back(ac);
        }
    }
    std::sort(result.begin(), result.end(),
        [](const AcupointCombination& a, const AcupointCombination& b) {
            return a.sample_count > b.sample_count;
        });
    return result;
}

std::string AcupunctureRecommender::itemset_key(const std::set<std::string>& s) {
    std::ostringstream oss;
    bool first = true;
    for (const auto& item : s) {
        if (!first) oss << "|";
        oss << item;
        first = false;
    }
    return oss.str();
}

bool AcupunctureRecommender::has_infix(const std::set<std::string>& set, const std::set<std::string>& subset) {
    for (const auto& s : subset) {
        if (set.find(s) == set.end()) return false;
    }
    return true;
}

} // namespace recommend
} // namespace tcm
