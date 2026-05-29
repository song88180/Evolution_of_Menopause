import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import Menopause_I as sim


def configure_test_simulation(seed=1, if_invasion=0, max_age=70):
    args = SimpleNamespace(
        out_dir="/tmp/menopause-test",
        sib_mortality=0,
        mat_mortality=0,
        lif_increase=0,
        epi_inherit=0,
        maternal_age_effect=0,
        if_invasion=if_invasion,
        interbirth_interval=3,
        k_s=1.5,
        x0_s=7,
        L_s=0.5,
        epi_h=0.05,
        max_age=max_age,
        U_curve_right_quadratic_term=0.004,
        U_curve_vertex_x=32.7,
        attenuation_cutoff=0,
        idx=1,
        seed=seed,
    )
    sim.configure_simulation(args)
    sim.People.created_people = 0


def make_person(sex, age=20, mother=None, father=None):
    return sim.People(
        sex=sex,
        Paternal_allele=sim.default_allele,
        Maternal_allele=sim.default_allele,
        gen_of_birth=0,
        age=age,
        Mother=mother,
        Father=father,
    )


def test_get_mortality_curve_marks_ages_after_max_age_as_terminal():
    ages, mortality = sim.get_mortality_curve(max_age=40)

    assert ages.tolist() == list(range(72))
    assert len(mortality) == 72
    assert np.all((mortality >= 0) & (mortality <= 1))
    assert mortality[0] == 0.15
    assert np.all(mortality[41:] == 1)


def test_survival_N_sib_is_bounded_and_decreases_with_more_siblings():
    configure_test_simulation()

    survival_values = [sim.survival_N_sib(n) for n in range(10)]

    assert all(0 <= value <= 1 for value in survival_values)
    assert survival_values == sorted(survival_values, reverse=True)


def test_get_attenuation_weight_tapers_between_ages_5_and_15():
    cutoff = 0.25

    assert sim.get_attenuation_weight(0, cutoff) == 1.0
    assert sim.get_attenuation_weight(5, cutoff) == 1.0
    assert sim.get_attenuation_weight(10, cutoff) == 0.625
    assert sim.get_attenuation_weight(15, cutoff) == cutoff
    assert sim.get_attenuation_weight(30, cutoff) == cutoff


def test_default_allele_matches_40_year_reproductive_lifespan_without_invasion():
    configure_test_simulation(if_invasion=0, max_age=70)

    assert sim.default_allele.index == 30
    assert 70 + sim.default_allele.effect == 40


def test_default_allele_matches_max_age_reproductive_lifespan_with_invasion():
    configure_test_simulation(if_invasion=1, max_age=55)

    assert sim.default_allele.index == 15
    assert 70 + sim.default_allele.effect == 55


def test_early_stop_succeeds_when_mean_stabilizes_below_threshold():
    configure_test_simulation()
    sim.early_stop_min_years = 5
    sim.early_stop_stability_years = 5
    sim.early_stop_stable_slope = 0.001

    history = [45.2, 45.2, 45.2, 45.2, 45.2, 45.2]

    assert sim.get_early_stop_status(5, history) == 'succeed'


def test_early_stop_fails_when_mean_is_stably_above_threshold():
    configure_test_simulation()
    sim.early_stop_min_years = 5
    sim.early_stop_stability_years = 5
    sim.early_stop_stable_slope = 0.001

    history = [46.3, 46.4, 46.4, 46.5, 46.5, 46.6]

    assert sim.get_early_stop_status(5, history) == 'failed'


def test_early_stop_waits_for_minimum_burn_in():
    configure_test_simulation()
    sim.early_stop_min_years = 10
    sim.early_stop_stability_years = 5
    sim.early_stop_stable_slope = 0.001

    history = [45.2, 45.2, 45.2, 45.2, 45.2, 45.2]

    assert sim.get_early_stop_status(5, history) is None


def test_population_step_cleans_dead_person_from_relatives_and_partner():
    configure_test_simulation(seed=2)
    pop = sim.Population(if_marriage=True)
    pop.mutation_rate = 0

    father = make_person(sex=1, age=30)
    mother = make_person(sex=0, age=30)
    father.Partner.append(mother)
    mother.Partner.append(father)

    child_a = make_person(sex=0, age=5, mother=mother, father=father)
    child_b = make_person(sex=1, age=5, mother=mother, father=father)
    child_a.sibling_list.append(child_b)
    child_b.sibling_list.append(child_a)
    child_a.N_Brother = 1
    child_b.N_Sister = 1

    mother.offspring_list.extend([child_a, child_b])
    father.offspring_list.extend([child_a, child_b])
    mother.N_daughters = 1
    mother.N_sons = 1
    father.N_daughters = 1
    father.N_sons = 1

    child_a.Partner.append(child_b)
    child_b.Partner.append(child_a)

    for person in [father, mother, child_b]:
        person.get_survival_rate = lambda: 1
    child_a.get_survival_rate = lambda: 0

    for person in [father, mother, child_a, child_b]:
        pop.Add_people(person)
    pop.update()

    pop.next_generation()

    assert child_a not in pop.Female_list
    assert child_a not in mother.offspring_list
    assert child_a not in father.offspring_list
    assert child_a not in child_b.sibling_list
    assert child_a not in child_b.Partner
    assert child_b.N_Sister == 0
    assert mother.N_daughters == 0
    assert father.N_daughters == 0
    assert pop.N_people_died == 1
