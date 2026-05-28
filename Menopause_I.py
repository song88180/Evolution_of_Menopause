import numpy as np
import numpy.random as nrand
import pandas as pd
import yaml
import scipy.stats
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return int(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return 1
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return 0
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Read simulation parameters")
parser.add_argument('--out-dir', type=str, required=True, help="Output directory")
parser.add_argument('--sib-mortality', type=str2bool, required=True, help="Number of siblings influences the mortality")
parser.add_argument('--mat-mortality', type=str2bool, required=False, help="Survival of the mother influences the mortality")
parser.add_argument('--lif-increase', type=str2bool, required=True, help="Gradually increase lifespan in evolution")
parser.add_argument('--epi-inherit', type=str2bool, required=True, help="Inherit epigenetic effect")
parser.add_argument('--maturity-effect', type=str2bool, required=True, help="Maturity effect on mortality")
parser.add_argument('--maternal-depletion-effect', type=str2bool, required=True, help="Maternal depletion effect on mortality")
parser.add_argument('--interbirth-interval', type=int, default=3, help="Interbirth interval")
parser.add_argument('--k-s', type=float, default=1.5, help="k in survival_N_sib function")
parser.add_argument('--x0-s', type=float, default=7, help="x0 in survival_N_sib function")
parser.add_argument('--L-s', type=float, default=0.5, help="L in survival_N_sib function")
parser.add_argument('--epi-h', type=float, default=0.05, help="heritability of the epigenetic effect")
parser.add_argument('--max-age', type=int, default=70, help="maximum lifespan")
parser.add_argument('--U-curve-left-quadratic-term', type=float, default=0.004, help="Quadratic term in U-curve")
parser.add_argument('--U-curve-right-quadratic-term', type=float, default=0.004, help="Quadratic term in U-curve")
parser.add_argument('--U-curve-vertex-x', type=float, default=32.7, help="Vertex x in U-curve")
parser.add_argument('--idx', type=int, required=True)

args = parser.parse_args()

out_folder = args.out_dir
Sibling_effect_mortality = args.sib_mortality
Maternal_effect_mortality = args.mat_mortality
Maturity_effect = args.maturity_effect
Maternal_depletion_effect = args.maternal_depletion_effect
interbirth_interval = args.interbirth_interval
if_lifespan = args.lif_increase
if_epi = args.epi_inherit
k_s = args.k_s
x0_s = args.x0_s
L_s = args.L_s
epi_h = args.epi_h
max_age = args.max_age
U_curve_left_quadratic_term = args.U_curve_left_quadratic_term
U_curve_right_quadratic_term = args.U_curve_right_quadratic_term
U_curve_vertex_x = args.U_curve_vertex_x
run_idx = args.idx


with open("options.yml",'r') as f:
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


def marriage_N_sib(N_sib):
    global k_m
    global x0_m
    global L_m
    y = -L_m / (1 + np.exp(k_m * (x0_m - N_sib))) + 1
    return y


# Pre-generate random values to reduce overhead in the long simulation loop.
random_list=nrand.random(size=10000000).tolist()
def get_random():
    global random_list
    if len(random_list) == 0:
        random_list=nrand.random(size=10000000).tolist()
    return random_list.pop()


class Allele:
    N=0
    def __init__(self,effect=0):
        self.index = Allele.N
        self.effect = effect
        Allele.N += 1

x,y = get_mortality_curve(max_age=max_age)
Primary_mortality_with_age_female = dict(zip(x, y))
Primary_mortality_with_age_male = Primary_mortality_with_age_female

allele_list = []
for i in range(36):
    # Higher allele indices lower menopause age by one year per effect unit.
    effect = -i
    allele_list.append(Allele(effect=effect))

default_allele = allele_list[30]

class People:
    destructed_people = 0
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
        self.mat_depletion_HR = 1
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
        # Mutate one inherited allele by one adjacent allele state.
        if get_random() < 0.5:
            mut_idx = self.Paternal_allele.index + nrand.choice([-1, 1])
            mut_idx = max(0, min(35, mut_idx))
            self.Paternal_allele = allele_list[mut_idx]

        else:
            mut_idx = self.Maternal_allele.index + nrand.choice([-1, 1])
            mut_idx = max(0, min(35, mut_idx))
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

        if self.Age <= 5:
            w = 1.0
        elif self.Age >= 15:
            w = 0.1
        else:
            w = 1.0 - 0.9 * (self.Age - 5) / 10.0

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

        if Maturity_effect:
            if (self.Mother is not None) and (self.Mother.Age < U_curve_vertex_x):
                hazard = hazard * np.exp(U_curve_left_quadratic_term*(self.Mother.Age - U_curve_vertex_x)**2)

        if Maternal_depletion_effect:
            hazard = hazard * self.mat_depletion_HR

        if self.Age <= 5:
            w = 1.0
        elif self.Age >= 15:
            w = 0.1
        else:
            w = 1.0 - 0.9 * (self.Age - 5) / 10.0

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
    

    
    def __del__(self):
        People.destructed_people += 1


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
        self.mutation_rate = 1/500
        
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
                
        nrand.shuffle(Males_to_marry)
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
            offspring.mat_depletion_HR = np.exp(U_curve_right_quadratic_term*(Female.Age - U_curve_vertex_x)**2)
        else:
            offspring.mat_depletion_HR = 1

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
        if self.N_male + self.N_female > 10000:
            self.pop_survival_rate = 5000/(self.N_male + self.N_female)
        else:
            self.pop_survival_rate = 1
        
        for sex,Pop_list in zip(['Male','Female'],[self.Male_list,self.Female_list]):
            for people in Pop_list:
                if get_random() < people.survival_rate - (1 - self.pop_survival_rate) :
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
                        assert people not in partner.sibling_list
                    
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


# Initialize both sexes across ages 0-9 with the default menopause allele.
Allele.N = 0
People.created_people = 0
People.destructed_people = 0


Pop = Population()
for sex in [0,1]:
    for age in range(10):
        for i in range(200):
            people = People(sex=sex,Paternal_allele=default_allele,Maternal_allele=default_allele,gen_of_birth=-age,age=age)
            Pop.Add_people(people)

people=None
Pop.MORTALITY_COEFFICIENT = argv['MORTALITY_COEFFICIENT']


allele_list_dict={i: [] for i in range(len(allele_list))}
N_list = []
sex_ratio_list = []
sex_ratio_at_birth_list = []

Menopause_age_list = []

start_age = 40
end_age = 70
N_years = 50000

for year in range(N_years + 1):
    print(f'{year}   ',end='\r')

    if year % 50 == 0:
        menopause_age_mean = Pop.get_mean_Menopause_age()
        print(menopause_age_mean)

    if if_lifespan:
        max_age = np.round((year / N_years) * (end_age - start_age) + start_age).astype(int)
        _age_, _mortality_ = get_mortality_curve(max_age)
        Primary_mortality_with_age_female = dict(zip(_age_, _mortality_))
        Primary_mortality_with_age_male = Primary_mortality_with_age_female

    Pop.reproduce()
    Pop.next_generation()

    # Record terminal allele frequencies and menopause ages for the summary.
    if year >= N_years - 100:
        N_pop = Pop.N_male + Pop.N_female
        allele_dict = Pop.get_allele_dict()
        for key in allele_list_dict.keys():
            if key in allele_dict:
                allele_list_dict[key].append(allele_dict[key] / N_pop / 2)
            else:
                allele_list_dict[key].append(0)
            
        Menopause_age_list.append(Pop.get_mean_Menopause_age())
    
    if People.created_people - People.destructed_people - (Pop.N_male+Pop.N_female) > 20000:
        break
    if (Pop.N_male + Pop.N_female) > 200000:
        break



label_dict = {i:-i for i in range(36)}

if Pop.N_male + Pop.N_female == 0:
    result_str = 'extinct\tNA\tNA\tNA'

else:
    Menopause_age_mean = np.mean(Menopause_age_list[-100:])
    
    AF_max = 0
    i_max = None
    for i in range(0, len(allele_list)):
        AF = np.mean(allele_list_dict[i][-100:])
        if AF > AF_max:
            AF_max = AF
            i_max = i

    if i_max != 0:
        result_str = f'succeed\t{Menopause_age_mean}\t{label_dict[i_max]}\t{AF_max}'
    else:
        result_str = f'failed\t{Menopause_age_mean}\t{label_dict[i_max]}\t{AF_max}'

with open(f'{out_folder}/MPSim_result_{Sibling_effect_mortality}{Maternal_effect_mortality}_{run_idx}.txt', 'w') as f:
    f.write(f'{Sibling_effect_mortality}\t{Maternal_effect_mortality}\t{k_s}\t{x0_s}\t{L_s}\t{max_age}\t' + result_str + '\n')
