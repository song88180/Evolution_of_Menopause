import numpy as np
import yaml
import argparse
import os


ALLELE_COUNT = 36
DEFAULT_ALLELE_INDEX = 30
INITIAL_AGE_CLASSES = 10
INITIAL_PEOPLE_PER_SEX_AGE = 200
MUTATION_RATE = 1 / 500
START_MAX_AGE = 40
END_MAX_AGE = 70
N_YEARS = 50000
TERMINAL_SUMMARY_YEARS = 100
DENSITY_CONTROL_THRESHOLD = 10000
DENSITY_CONTROL_TARGET = 5000
MAX_RETAINED_DEAD_REFERENCES = 20000
MAX_POPULATION_SIZE = 200000
MENOPAUSE_EVOLUTION_AGE_THRESHOLD = 46
EARLY_STOP_MIN_YEARS = 1000
EARLY_STOP_STABILITY_YEARS = 500
EARLY_STOP_STABLE_SLOPE = 0.001


def str2bool(v):
    if isinstance(v, bool):
        return int(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return 1
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return 0
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def attenuation_cutoff_type(v):
    v = float(v)
    if 0 <= v <= 1:
        return v
    raise argparse.ArgumentTypeError('attenuation_cutoff must be between 0 and 1, inclusive.')

parser = argparse.ArgumentParser(description="Read simulation parameters")
parser.add_argument('--out-dir', type=str, required=True, help="Output directory")
parser.add_argument('--sib-mortality', type=str2bool, required=True, help="Number of siblings influences the mortality")
parser.add_argument('--mat-mortality', type=str2bool, default=False, required=False, help="Survival of the mother influences the mortality")
parser.add_argument('--lif-increase', type=str2bool, required=True, help="Gradually increase lifespan in evolution")
parser.add_argument('--epi-inherit', type=str2bool, required=True, help="Inherit epigenetic effect")
parser.add_argument('--maternal-age-effect', type=str2bool, required=True, help="Maternal age effect on mortality")
parser.add_argument('--if-invasion', type=str2bool, default=False, help="Use max_age as the initial reproductive lifespan")
parser.add_argument('--interbirth-interval', type=int, default=3, help="Interbirth interval")
parser.add_argument('--k-s', type=float, default=1.5, help="k in survival_N_sib function")
parser.add_argument('--x0-s', type=float, default=7, help="x0 in survival_N_sib function")
parser.add_argument('--L-s', type=float, default=0.5, help="L in survival_N_sib function")
parser.add_argument('--epi-h', type=float, default=0.05, help="heritability of the epigenetic effect")
parser.add_argument('--max-age', type=int, default=70, help="maximum lifespan")
parser.add_argument('--U-curve-right-quadratic-term', type=float, default=0.004, help="Quadratic term in U-curve")
parser.add_argument('--U-curve-vertex-x', type=float, default=32.7, help="Vertex x in U-curve")
parser.add_argument('--attenuation_cutoff', '--attenuation-cutoff', type=attenuation_cutoff_type, default=0, help="Attenuation weight after age 15")
parser.add_argument('--idx', type=int, required=True)
parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducible simulation runs")
parser.add_argument('--early-stop-min-years', type=int, default=EARLY_STOP_MIN_YEARS, help="Minimum years before checking early-stop criteria")
parser.add_argument('--early-stop-stability-years', type=int, default=EARLY_STOP_STABILITY_YEARS, help="Rolling years used to decide whether menopause age is stable")
parser.add_argument('--early-stop-stable-slope', type=float, default=EARLY_STOP_STABLE_SLOPE, help="Maximum absolute yearly slope treated as stable")


out_folder = None
Sibling_effect_mortality = 0
Maternal_effect_mortality = False
Maternal_age_effect = 0
interbirth_interval = 3
if_lifespan = 0
if_epi = 0
if_invasion = False
k_s = 1.5
x0_s = 7
L_s = 0.5
epi_h = 0.05
max_age = 70
U_curve_right_quadratic_term = 0.004
U_curve_vertex_x = 32.7
attenuation_cutoff = 0
run_idx = None
rng = np.random.default_rng()
early_stop_min_years = EARLY_STOP_MIN_YEARS
early_stop_stability_years = EARLY_STOP_STABILITY_YEARS
early_stop_stable_slope = EARLY_STOP_STABLE_SLOPE


options_path = os.path.join(os.path.dirname(__file__), "options.yml")
with open(options_path,'r') as f:
    argv = yaml.load(f,Loader=yaml.FullLoader)

# Baseline age schedules come from options.yml; mortality is generated below.
Primary_reproduction_rate_with_age_female = argv['Primary_reproduction_rate_with_age_female']
Primary_reproduction_rate_with_age_male = argv['Primary_reproduction_rate_with_age_male']
Primary_marriage_rate_with_age_female = argv['Primary_marriage_rate_with_age_female']
Primary_marriage_rate_with_age_male = argv['Primary_marriage_rate_with_age_male']


def get_mortality_curve(max_age=70):
    # Infant and childhood mortality decline, then adult mortality rises exponentially.
    x = np.linspace(0,71,72).astype(int)
    
    A = 0.07
    B = -0.3
    x1 = x[1:11]
    y1 = A*np.exp(B*x1) + (0.001 - A*np.exp(B*11))
    
    x2 = x[11:]
    y2 = 0.001*np.exp(0.07*(59/(max_age-11))*(x2-11))
    
    y = np.array([0.15] + list(y1) + list(y2))
    
    y[max_age+1:] = 1
    return x, y


def survival_N_sib(N_sib):
    # More young siblings reduce survival through a saturating response.
    global k_s
    global x0_s
    global L_s
    if N_sib == x0_s:
        y = -k_s / L_s + 1 - k_s*x0_s/(1-np.exp(L_s*x0_s))
    else:
        y = -k_s*(N_sib - x0_s) / (1 - np.exp(-L_s * (N_sib - x0_s))) + 1 - k_s*x0_s/(1-np.exp(L_s*x0_s))
    if y < 0:
        y = 0
    return y


def get_attenuation_weight(age, attenuation_cutoff):
    if age <= 5:
        return 1.0
    if age >= 15:
        return attenuation_cutoff
    return 1.0 - (1.0 - attenuation_cutoff) * (age - 5) / 10.0


def marriage_N_sib(N_sib):
    y = -L_s / (1 + np.exp(k_s * (x0_s - N_sib))) + 1
    return y


def get_random():
    return rng.random()


def get_default_allele_index(if_invasion, max_age):
    if if_invasion:
        default_index = int(70 - max_age)
    else:
        default_index = DEFAULT_ALLELE_INDEX

    if not 0 <= default_index < ALLELE_COUNT:
        raise ValueError(
            f"Initial reproductive lifespan requires allele index {default_index}, "
            f"but valid indices are 0 to {ALLELE_COUNT - 1}."
        )
    return default_index


class Allele:
    N=0
    def __init__(self,effect=0):
        self.index = Allele.N
        self.effect = effect
        Allele.N += 1

x,y = get_mortality_curve(max_age=max_age)
Primary_mortality_with_age_female = dict(zip(x, y))
Primary_mortality_with_age_male = Primary_mortality_with_age_female


def configure_simulation(args):
    global out_folder
    global Sibling_effect_mortality
    global Maternal_effect_mortality
    global Maternal_age_effect
    global interbirth_interval
    global if_lifespan
    global if_epi
    global if_invasion
    global k_s
    global x0_s
    global L_s
    global epi_h
    global max_age
    global U_curve_right_quadratic_term
    global U_curve_vertex_x
    global attenuation_cutoff
    global run_idx
    global rng
    global early_stop_min_years
    global early_stop_stability_years
    global early_stop_stable_slope
    global default_allele
    global Primary_mortality_with_age_female
    global Primary_mortality_with_age_male

    out_folder = args.out_dir
    Sibling_effect_mortality = args.sib_mortality
    Maternal_effect_mortality = args.mat_mortality
    Maternal_age_effect = args.maternal_age_effect
    interbirth_interval = args.interbirth_interval
    if_lifespan = args.lif_increase
    if_epi = args.epi_inherit
    if_invasion = args.if_invasion
    k_s = args.k_s
    x0_s = args.x0_s
    L_s = args.L_s
    epi_h = args.epi_h
    max_age = args.max_age
    U_curve_right_quadratic_term = args.U_curve_right_quadratic_term
    U_curve_vertex_x = args.U_curve_vertex_x
    attenuation_cutoff = args.attenuation_cutoff
    run_idx = args.idx
    rng = np.random.default_rng(args.seed)
    early_stop_min_years = getattr(args, 'early_stop_min_years', EARLY_STOP_MIN_YEARS)
    early_stop_stability_years = getattr(args, 'early_stop_stability_years', EARLY_STOP_STABILITY_YEARS)
    early_stop_stable_slope = getattr(args, 'early_stop_stable_slope', EARLY_STOP_STABLE_SLOPE)

    x, y = get_mortality_curve(max_age=max_age)
    Primary_mortality_with_age_female = dict(zip(x, y))
    Primary_mortality_with_age_male = Primary_mortality_with_age_female
    default_allele = allele_list[get_default_allele_index(if_invasion, max_age)]


allele_list = []
for i in range(ALLELE_COUNT):
    # Higher allele indices lower menopause age by one year per effect unit.
    effect = -i
    allele_list.append(Allele(effect=effect))

default_allele = allele_list[DEFAULT_ALLELE_INDEX]

class People:
    created_people = 0
    Male_age_cutoff = 15
    Female_age_cutoff = 15
    def __init__(self,sex,Paternal_allele,Maternal_allele,gen_of_birth,age=0,
                 N_sons=0,N_daughters=0,N_brothers=0,N_sisters=0,Mother=None,Father=None):
        self.Gen_of_birth = gen_of_birth
        self.Age = age
        self.Sex = sex
        self.Paternal_allele = Paternal_allele
        self.Maternal_allele = Maternal_allele
        self.Resource = 1
        self.Menopause_age = 70 + np.mean([self.Paternal_allele.effect, self.Maternal_allele.effect])
        self.N_sons = N_sons
        self.N_daughters = N_daughters
        self.N_birth = 0
        self.offspring_list = []
        self.Mother = Mother
        self.Father = Father
        self.Partner = []
        self.N_Brother = 0
        self.N_Sister = 0
        self.sibling_list = []
        self.N_young_sib_list = []
        self.survival_rate = 1
        self.maternal_age_HR = 1
        self.mating_willingness = self.get_mating_willingness()
        self.marry_willingness = self.get_marry_willingness()
        People.created_people += 1
        self.epi_survival_rate = 1

        if if_epi:
            if self.Mother is not None:
                survival_rate_sib_eff = np.mean([survival_N_sib(N_young_sib) for N_young_sib in self.Mother.N_young_sib_list])
                self.epi_survival_rate = 1 - (1 - survival_rate_sib_eff) * epi_h
            else:
                self.epi_survival_rate = 1
    
    def update(self):
        self.survival_rate = self.get_survival_rate()
        self.mating_willingness = self.get_mating_willingness()
        self.marry_willingness = self.get_marry_willingness()
        
    def mutate(self):
        # Mutate one inherited allele by a step of -2, -1, 1, or 2 allele states.
        if get_random() < 0.5:
            mut_idx = self.Paternal_allele.index + rng.choice([-2, -1, 1, 2])
            mut_idx = max(0, min(ALLELE_COUNT - 1, mut_idx))
            self.Paternal_allele = allele_list[mut_idx]

        else:
            mut_idx = self.Maternal_allele.index + rng.choice([-2, -1, 1, 2])
            mut_idx = max(0, min(ALLELE_COUNT - 1, mut_idx))
            self.Maternal_allele = allele_list[mut_idx]
            
        self.Menopause_age = 70 + np.mean([self.Paternal_allele.effect, self.Maternal_allele.effect])
        
    def update_N_young_sib_list(self):        
        # Record the number of dependent maternal siblings at this age.
        N_young_sib = 0
        for sib in self.sibling_list:
            if (sib.Sex == 0) and (sib.Age < People.Female_age_cutoff):
                N_young_sib += 1
            elif (sib.Sex == 1) and (sib.Age < People.Male_age_cutoff):
                N_young_sib += 1
                
        self.N_young_sib_list.append(N_young_sib)
    
    def get_sibling_effect_mortality(self):

        if len(self.N_young_sib_list) > 0:
            survival_rate_multiplier = np.mean([survival_N_sib(N_young_sib) for N_young_sib in self.N_young_sib_list])
        else:
            survival_rate_multiplier = 1

        if if_epi:
            survival_rate_multiplier = survival_rate_multiplier * self.epi_survival_rate

        w = get_attenuation_weight(self.Age, attenuation_cutoff)

        # Sibling pressure is strongest in early childhood and tapers after age 5.
        survival_rate_multiplier = 1 - (1 - survival_rate_multiplier) * w

        return survival_rate_multiplier
    
    def get_survival_rate(self):
        if self.Sex == 0:
            primary_mortality = Primary_mortality_with_age_female[self.Age]
        else:
            primary_mortality = Primary_mortality_with_age_male[self.Age]
        
        _survival_rate = (1 - primary_mortality)
        
        if Sibling_effect_mortality:
            survival_rate_multiplier = self.get_sibling_effect_mortality()
            _survival_rate = _survival_rate * survival_rate_multiplier

        hazard = 1

        if Maternal_effect_mortality:
            if (self.Mother is None) and (self.Age <= 10):
                hazard = hazard * 10

        if Maternal_age_effect:
            hazard = hazard * self.maternal_age_HR

        w = get_attenuation_weight(self.Age, attenuation_cutoff)

        # Maternal hazards are age-weighted to matter most for young children.
        hazard = 1 + w * (hazard - 1)

        _survival_rate = 1 - (1 - _survival_rate) * hazard

        return max(0,min(1,_survival_rate))
    
    
    def get_sibling_effect_marriage(self):
        if len(self.N_young_sib_list) > 0:
            mating_willingness_multiplier = np.mean([marriage_N_sib(N_young_sib) for N_young_sib in self.N_young_sib_list])
        else:
            mating_willingness_multiplier = 1
        
        return mating_willingness_multiplier
    
    def get_marry_willingness(self):
        if self.Sex == 0:
            _marry_willingness = Primary_marriage_rate_with_age_female[self.Age]     
        else:
            _marry_willingness = Primary_marriage_rate_with_age_male[self.Age]
        
        if argv['Sibling_effect_marriage']:
            marry_willingness_multiplier = self.get_sibling_effect_marriage()
            _marry_willingness = _marry_willingness * marry_willingness_multiplier
            
        return max(0, min(1, _marry_willingness))
    
    
    def get_mating_willingness(self):
        if self.Sex == 0:
            _mating_willingness = Primary_reproduction_rate_with_age_female[self.Age]
            
            if self.Age >= self.Menopause_age:
                _mating_willingness = 0
            
            elif self.N_daughters + self.N_sons > 0:
                child_age_min = np.min([child.Age for child in self.offspring_list])
                if child_age_min < interbirth_interval:
                    _mating_willingness = 0
                    
        else:
            _mating_willingness = Primary_reproduction_rate_with_age_male[self.Age]
            
        return max(0, min(1, _mating_willingness))
    


class Population:
    def __init__(self, if_marriage=False):
        self.N_male = 0
        self.N_female = 0
        self.Male_list = []
        self.Female_list = []
        self.Current_generation = 0
        self.MORTALITY_COEFFICIENT = 2e-12
        self.pop_survival_rate = 1
        self.update()
        self.N_people_died = 0
        self.if_marriage = if_marriage
        self.mutation_rate = MUTATION_RATE
        
    def Add_people(self, people):
        if people.Sex == 0:
            self.Female_list.append(people)
        else:
            self.Male_list.append(people)
    
    def get_mean_Menopause_age(self):
        Menopause_age_list = []
        for people in self.Female_list:
            Menopause_age_list.append(people.Menopause_age)
        return np.mean(Menopause_age_list)
    
    def marry(self):
        # Males can retain multiple partners; females only enter if unpartnered.
        Males_to_marry = []
        Females_to_marry = []
        
        for people in self.Male_list:
            if get_random() < people.marry_willingness:
                Males_to_marry.append(people)
                
        for people in self.Female_list:
            if (len(people.Partner) == 0) and (get_random() < people.marry_willingness):
                Females_to_marry.append(people)
                
        rng.shuffle(Males_to_marry)
        for i in range(min(len(Males_to_marry),len(Females_to_marry))):
            Males_to_marry[i].Partner.append(Females_to_marry[i])
            Females_to_marry[i].Partner.append(Males_to_marry[i])
            
    def mate2people(self, Male, Female):
        
        if get_random() < 0.5:
            sex = 1
            Male.N_sons += 1
            Female.N_sons += 1
        else:
            sex = 0
            Male.N_daughters += 1
            Female.N_daughters += 1
        
        Female.N_birth += 1
        
        Paternal_allele = Male.Paternal_allele if get_random() < 0.5 else Male.Maternal_allele
        Maternal_allele = Female.Paternal_allele if get_random() < 0.5 else Female.Maternal_allele
        
        offspring = People(sex,Paternal_allele,Maternal_allele,
                           gen_of_birth=self.Current_generation,age=0,
                           Mother=Female,Father=Male)


        if Female.Age > U_curve_vertex_x:
            offspring.maternal_age_HR = np.exp(U_curve_right_quadratic_term*(Female.Age - U_curve_vertex_x)**2)
        else:
            offspring.maternal_age_HR = 1

        # Sibling effects are based on maternal siblings.
        for sibling in Female.offspring_list:
            if sex == 0:
                sibling.N_Sister += 1
            else:
                sibling.N_Brother += 1
            sibling.sibling_list.append(offspring)
            offspring.sibling_list.append(sibling)
        
        Male.offspring_list.append(offspring)
        Female.offspring_list.append(offspring)
        
        if sex == 0:
            self.Female_list.append(offspring)
        else:
            self.Male_list.append(offspring)
        
    def reproduce(self):
        for people in self.Female_list:
            if (len(people.Partner) > 0) and (get_random() < people.mating_willingness):
                self.mate2people(people.Partner[0], people)
        
    def next_generation(self):
        Male_list_new = []
        Female_list_new = []
        
        for people in [*self.Male_list, *self.Female_list]:
            if (people.Sex == 0) and (people.Age < People.Female_age_cutoff):
                people.update_N_young_sib_list()
            elif (people.Sex == 1) and (people.Age < People.Male_age_cutoff):
                people.update_N_young_sib_list()
                
            if not self.if_marriage:
                people.Partner = []
                
            if get_random() < self.mutation_rate:
                people.mutate()
            
            people.survival_rate = people.get_survival_rate()
        
        # Apply density control when the population grows too large.
        if self.N_male + self.N_female > DENSITY_CONTROL_THRESHOLD:
            self.pop_survival_rate = DENSITY_CONTROL_TARGET/(self.N_male + self.N_female)
        else:
            self.pop_survival_rate = 1
        
        for sex,Pop_list in zip(['Male','Female'],[self.Male_list,self.Female_list]):
            for people in Pop_list:
                if get_random() < people.survival_rate * self.pop_survival_rate:
                    people.Age += 1
                    if sex == 'Male':
                        Male_list_new.append(people)
                    else:
                        Female_list_new.append(people)
                else:
                    # Remove references to dead individuals from relatives and partners.
                    if people.Mother is not None:
                        if sex == 'Male':
                            people.Mother.N_sons -= 1
                        else:
                            people.Mother.N_daughters -= 1
                        people.Mother.offspring_list.remove(people)
                        assert people not in people.Mother.offspring_list
                        
                    if people.Father is not None:
                        if sex == 'Male':
                            people.Father.N_sons -= 1
                        else:
                            people.Father.N_daughters -= 1
                        people.Father.offspring_list.remove(people)
                        assert people not in people.Father.offspring_list
                        
                    for offspring in people.offspring_list:
                        if sex == 'Male':
                            offspring.Father = None
                        else:
                            offspring.Mother = None
                    
                    for sibling in people.sibling_list:
                        if sex == 'Male':
                            sibling.N_Brother -= 1
                        else:
                            sibling.N_Sister -= 1
                        sibling.sibling_list.remove(people)
                        assert people not in sibling.sibling_list
                    
                    for partner in people.Partner:
                        partner.Partner.remove(people)
                        assert people not in partner.Partner
                    
                    self.N_people_died += 1
                        
        self.Male_list = Male_list_new
        self.Female_list = Female_list_new
        
        for people in [*self.Male_list, *self.Female_list]:
            people.mating_willingness = people.get_mating_willingness()
            people.marry_willingness = people.get_marry_willingness()
        
        self.marry()
        
        self.Current_generation += 1
        
        self.update()
    
    def get_allele_dict(self):
        allele_dict = {}
        for Pop in [self.Male_list,self.Female_list]:
            for people in Pop:
                if people.Paternal_allele.index in allele_dict:
                    allele_dict[people.Paternal_allele.index] += 1
                else:
                    allele_dict[people.Paternal_allele.index] = 1
                if people.Maternal_allele.index in allele_dict:
                    allele_dict[people.Maternal_allele.index] += 1
                else:
                    allele_dict[people.Maternal_allele.index] = 1
        return allele_dict
    
    def update(self):
        self.N_female = len(self.Female_list)
        self.N_male = len(self.Male_list)


def initialize_population():
    People.created_people = 0

    Pop = Population()
    for sex in [0,1]:
        for age in range(INITIAL_AGE_CLASSES):
            for i in range(INITIAL_PEOPLE_PER_SEX_AGE):
                people = People(sex=sex,Paternal_allele=default_allele,Maternal_allele=default_allele,gen_of_birth=-age,age=age)
                Pop.Add_people(people)

    Pop.MORTALITY_COEFFICIENT = argv['MORTALITY_COEFFICIENT']
    return Pop


def record_population_summary(Pop, allele_list_dict, Menopause_age_list):
    N_pop = Pop.N_male + Pop.N_female
    if N_pop == 0:
        return

    allele_dict = Pop.get_allele_dict()
    for key in allele_list_dict.keys():
        if key in allele_dict:
            allele_list_dict[key].append(allele_dict[key] / N_pop / 2)
        else:
            allele_list_dict[key].append(0)

    Menopause_age_list.append(Pop.get_mean_Menopause_age())


def get_recent_menopause_trend(menopause_age_history, window_years):
    if len(menopause_age_history) < window_years:
        return None

    recent_values = np.array(menopause_age_history[-window_years:])
    if np.any(np.isnan(recent_values)):
        return None

    years = np.arange(window_years)
    slope, _ = np.polyfit(years, recent_values, 1)
    return {
        'mean': np.mean(recent_values),
        'min': np.min(recent_values),
        'max': np.max(recent_values),
        'slope': slope,
    }


def get_early_stop_status(year, menopause_age_history):
    if year < early_stop_min_years:
        return None

    trend = get_recent_menopause_trend(menopause_age_history, early_stop_stability_years)
    if trend is None:
        return None

    if (
        abs(trend['slope']) <= early_stop_stable_slope
        and trend['mean'] < MENOPAUSE_EVOLUTION_AGE_THRESHOLD
    ):
        return 'succeed'

    if (
        trend['min'] > MENOPAUSE_EVOLUTION_AGE_THRESHOLD
        and trend['slope'] >= -early_stop_stable_slope
    ):
        return 'failed'

    return None


def run_simulation():
    global max_age
    global Primary_mortality_with_age_female
    global Primary_mortality_with_age_male

    Pop = initialize_population()
    allele_list_dict={i: [] for i in range(len(allele_list))}
    Menopause_age_list = []
    menopause_age_history = []
    early_stop_status = None
    menopause_age_report_override = None

    for year in range(N_YEARS + 1):
        print(f'{year}   ',end='\r')

        menopause_age_mean = Pop.get_mean_Menopause_age()
        menopause_age_history.append(menopause_age_mean)

        if year % 50 == 0:
            print(menopause_age_mean)

        if if_lifespan:
            max_age = np.round((year / N_YEARS) * (END_MAX_AGE - START_MAX_AGE) + START_MAX_AGE).astype(int)
            _age_, _mortality_ = get_mortality_curve(max_age)
            Primary_mortality_with_age_female = dict(zip(_age_, _mortality_))
            Primary_mortality_with_age_male = Primary_mortality_with_age_female

        Pop.reproduce()
        Pop.next_generation()

        # Keep a rolling terminal summary once the burn-in period has passed.
        if year >= max(0, early_stop_min_years - TERMINAL_SUMMARY_YEARS):
            record_population_summary(Pop, allele_list_dict, Menopause_age_list)

        if year % 50 == 0:
            early_stop_status = get_early_stop_status(year, menopause_age_history)
            if early_stop_status is not None:
                if early_stop_status == 'failed':
                    menopause_age_report_override = f'>{MENOPAUSE_EVOLUTION_AGE_THRESHOLD}'
                break
        
        if People.created_people - Pop.N_people_died - (Pop.N_male+Pop.N_female) > MAX_RETAINED_DEAD_REFERENCES:
            break
        if (Pop.N_male + Pop.N_female) > MAX_POPULATION_SIZE:
            break



    label_dict = {i:-i for i in range(ALLELE_COUNT)}

    if Pop.N_male + Pop.N_female == 0:
        result_str = 'extinct\tNA\tNA\tNA'

    else:
        if len(Menopause_age_list) == 0:
            record_population_summary(Pop, allele_list_dict, Menopause_age_list)

        Menopause_age_mean = np.mean(Menopause_age_list[-100:])
        
        AF_max = 0
        i_max = None
        for i in range(0, len(allele_list)):
            AF = np.mean(allele_list_dict[i][-100:])
            if AF > AF_max:
                AF_max = AF
                i_max = i

        Menopause_age_report = menopause_age_report_override or Menopause_age_mean

        if early_stop_status is not None:
            result_str = f'{early_stop_status}\t{Menopause_age_report}\t{label_dict[i_max]}\t{AF_max}'
        elif Menopause_age_mean < MENOPAUSE_EVOLUTION_AGE_THRESHOLD:
            result_str = f'succeed\t{Menopause_age_mean}\t{label_dict[i_max]}\t{AF_max}'
        else:
            result_str = f'failed\t{Menopause_age_mean}\t{label_dict[i_max]}\t{AF_max}'

    os.makedirs(out_folder, exist_ok=True)

    with open(f'{out_folder}/MPSim_result_{Sibling_effect_mortality}{Maternal_effect_mortality}_{run_idx}.txt', 'w') as f:
        f.write(f'{Sibling_effect_mortality}\t{Maternal_effect_mortality}\t{k_s}\t{x0_s}\t{L_s}\t{max_age}\t' + result_str + '\n')


def main():
    args = parser.parse_args()
    configure_simulation(args)
    run_simulation()


if __name__ == '__main__':
    main()
