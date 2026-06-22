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
        linear_effect=False,
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


def test_linear_sibling_effect_uses_only_k_s_and_is_bounded_at_zero():
    configure_test_simulation()
    sim.linear_effect = True
    sim.k_s = 0.25

    assert [sim.survival_N_sib(n) for n in range(6)] == [1.0, 0.75, 0.5, 0.25, 0, 0]


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

    assert sim.get_early_stop_status(5, history) == ('succeed', None)


def test_early_stop_fails_when_mean_is_stably_above_threshold():
    configure_test_simulation()
    sim.early_stop_min_years = 5
    sim.early_stop_stability_years = 5
    sim.early_stop_stable_slope = 0.001

    history = [46.5, 46.5, 46.5, 46.5, 46.5, 46.5]

    assert sim.get_early_stop_status(5, history) == ('failed', None)


def test_early_stop_fails_when_trend_rises_above_threshold():
    configure_test_simulation()
    sim.early_stop_min_years = 5
    sim.early_stop_stability_years = 5
    sim.early_stop_stable_slope = 0.001

    history = [46.3, 46.4, 46.4, 46.5, 46.5, 46.6]

    assert sim.get_early_stop_status(5, history) == ('failed', '>46')


def test_early_stop_succeeds_when_trend_declines_below_start_max_age():
    configure_test_simulation()
    sim.early_stop_min_years = 5
    sim.early_stop_stability_years = 5
    sim.early_stop_stable_slope = 0.001

    history = [39.0, 38.5, 38.0, 37.5, 37.0, 36.5]

    assert sim.get_early_stop_status(5, history) == ('succeed', '<39')


def test_early_stop_waits_for_minimum_burn_in():
    configure_test_simulation()
    sim.early_stop_min_years = 10
    sim.early_stop_stability_years = 5
    sim.early_stop_stable_slope = 0.001

    history = [45.2, 45.2, 45.2, 45.2, 45.2, 45.2]

    assert sim.get_early_stop_status(5, history) is None


def test_output_summary_uses_requested_order_and_na_for_irrelevant_parameters():
    configure_test_simulation()
    pop = sim.Population()
    female = make_person(sex=0, age=30)
    male = make_person(sex=1, age=30)
    pop.Add_people(female)
    pop.Add_people(male)
    pop.update()

    allele_list_dict = {i: [] for i in range(len(sim.allele_list))}
    allele_list_dict[sim.default_allele.index] = [0.6, 0.7, 0.8]
    menopause_age_list = [44.0, 45.0, 46.0]

    summary = sim.build_output_summary(
        status='succeed',
        final_year=12,
        stop_reason='completed',
        Pop=pop,
        allele_list_dict=allele_list_dict,
        Menopause_age_list=menopause_age_list,
        dominant_allele_index=sim.default_allele.index,
        dominant_allele_frequency=0.7,
        menopause_age_report=45.0,
    )

    assert list(summary.keys()) == [
        'run_idx',
        'sib_mortality',
        'maternal_age_effect',
        'mat_mortality',
        'attenuation_cutoff',
        'if_epi',
        'if_invasion',
        'linear_effect',
        'k_s',
        'x0_s',
        'L_s',
        'max_age',
        'maternal_age_effect_quadratic_term',
        'maternal_age_effect_vertex_x',
        'epi_h',
        'status',
        'stop_reason',
        'final_year',
        'menopause_age_report',
        'menopause_age_mean',
        'dominant_allele',
        'dominant_allele_frequency',
    ]
    assert summary['run_idx'] == sim.run_idx
    assert summary['sib_mortality'] == 0
    assert summary['maternal_age_effect'] == 0
    assert summary['mat_mortality'] == 0
    assert summary['attenuation_cutoff'] is None
    assert summary['if_epi'] == 0
    assert summary['if_invasion'] == 0
    assert summary['linear_effect'] is None
    assert summary['k_s'] is None
    assert summary['x0_s'] is None
    assert summary['L_s'] is None
    assert summary['menopause_age_mean'] == 45.0
    assert summary['maternal_age_effect_quadratic_term'] is None
    assert summary['maternal_age_effect_vertex_x'] is None
    assert summary['epi_h'] is None
    assert summary['dominant_allele'] == -sim.default_allele.index
    assert summary['dominant_allele_frequency'] == 0.7


def test_output_summary_keeps_relevant_parameters():
    configure_test_simulation(if_invasion=1, max_age=55)
    sim.Sibling_effect_mortality = 1
    sim.Maternal_age_effect = 1
    sim.Maternal_effect_mortality = 1
    sim.if_epi = 1

    pop = sim.Population()
    pop.Add_people(make_person(sex=0, age=30))
    pop.Add_people(make_person(sex=1, age=30))
    pop.update()

    allele_list_dict = {i: [] for i in range(len(sim.allele_list))}
    allele_list_dict[sim.default_allele.index] = [0.7]

    summary = sim.build_output_summary(
        status='succeed',
        final_year=12,
        stop_reason='completed',
        Pop=pop,
        allele_list_dict=allele_list_dict,
        Menopause_age_list=[45.0],
        dominant_allele_index=sim.default_allele.index,
        dominant_allele_frequency=0.7,
        menopause_age_report=45.0,
    )

    assert summary['attenuation_cutoff'] == sim.attenuation_cutoff
    assert summary['k_s'] == sim.k_s
    assert summary['x0_s'] == sim.x0_s
    assert summary['L_s'] == sim.L_s
    assert summary['maternal_age_effect_quadratic_term'] == sim.U_curve_right_quadratic_term
    assert summary['maternal_age_effect_vertex_x'] == sim.U_curve_vertex_x
    assert summary['epi_h'] == sim.epi_h


def test_output_summary_omits_saturating_parameters_for_linear_effect():
    configure_test_simulation()
    sim.Sibling_effect_mortality = 1
    sim.linear_effect = True

    summary = sim.build_output_summary(
        status='succeed',
        final_year=12,
        stop_reason='completed',
        Pop=sim.Population(),
        allele_list_dict={i: [] for i in range(len(sim.allele_list))},
        Menopause_age_list=[45.0],
        dominant_allele_index=None,
        dominant_allele_frequency=None,
        menopause_age_report=45.0,
    )

    assert summary['linear_effect'] is True
    assert summary['k_s'] == sim.k_s
    assert summary['x0_s'] is None
    assert summary['L_s'] is None


def test_write_output_summary_includes_parameter_names_as_header(tmp_path):
    output_path = tmp_path / "summary.txt"
    summary = {
        'sib_mortality': 1,
        'mat_mortality': 0,
        'k_s': 0.3,
        'status': 'succeed',
    }

    sim.write_output_summary(output_path, summary)

    lines = output_path.read_text().splitlines()
    assert lines[0] == 'sib_mortality\tmat_mortality\tk_s\tstatus'
    assert lines[1] == '1\t0\t0.3\tsucceed'


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
