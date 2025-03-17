import numpy as np
import numpy.random as nrand
import pandas as pd
import yaml
import scipy.stats
import pickle
import argparse

parser = argparse.ArgumentParser(description="Read simulation parameters")

parser.add_argument('--meno-age', type=int, required=True)

args = parser.parse_args()

meno_age = args.meno_age

def get_mortality_curve(max_age=70):

    fold_change = (70 - 11) / (max_age - 11)
    x = np.linspace(0,71,72).astype(int)

    A = 0.07
    B = -0.3
    x1 = x[1:11]
    y1 = A*np.exp(B*x1) + (0.001 - A*np.exp(B*11))

    x2 = x[11:]
    y2 = 0.001*np.exp(0.07*(x2-11))

    y = np.array([0.15] + list(y1) + list(y2))

    y[max_age+1:] = 1
    return x, y

with open("options.yml",'r') as f:
    argv = yaml.load(f,Loader=yaml.FullLoader)
Primary_reproduction_rate_with_age_male = argv['Primary_reproduction_rate_with_age_male']
Primary_reproduction_rate_with_age_female = argv['Primary_reproduction_rate_with_age_female']
Primary_marriage_rate_with_age_female = argv['Primary_marriage_rate_with_age_female']
Primary_marriage_rate_with_age_male = argv['Primary_marriage_rate_with_age_male']
x,y = get_mortality_curve(max_age=70) 
Primary_mortality_with_age_female = dict(zip(x, y))
Primary_mortality_with_age_male = Primary_mortality_with_age_female

if_epi = False
epi_h = 0.05

k_s = 0.2
x0_s = 4
L_s = 3

def survival_N_sib(N_sib):
    global k_s
    global x0_s
    global L_s
    if N_sib == x0_s:
        y = -k_s / L_s + 1 - k_s*x0_s/(1-np.exp(L_s*x0_s))
    else:
        y = -k_s*(N_sib - x0_s) / (1 - np.exp(-L_s * (N_sib - x0_s))) + 1 - k_s*x0_s/(1-np.exp(L_s*x0_s))
    return max(y, 0)

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

class People:
    destructed_people = 0
    created_people = 0
    Male_age_cutoff = 25
    Female_age_cutoff = 19
    def __init__(self,sex,Paternal_allele,Maternal_allele,gen_of_birth,age=0,
                 N_sons=0,N_daughters=0,N_brothers=0,N_sisters=0,Mother=None,Father=None):
        self.Gen_of_birth = gen_of_birth
        self.Age = age
        self.Sex = sex #0-female, 1-male
        self.Paternal_allele = Paternal_allele
        self.Maternal_allele = Maternal_allele
        self.Resource = 1
        self.Menopause_age = 70 + np.min([self.Paternal_allele.effect, self.Maternal_allele.effect])
        self.N_sons = N_sons
        self.N_daughters = N_daughters
        self.N_birth = 0
        self.N_mature_children = 0
        self.offspring_list = []
        self.Mother = Mother
        self.Father = Father
        self.Partner = []
        self.N_Brother = 0
        self.N_Sister = 0
        self.sibling_list = []
        self.N_young_sib_list = []
        self.survival_rate = 1
        self.mating_willingness = self.get_mating_willingness()
        self.marry_willingness = self.get_marry_willingness()
        #self.uid = self.get_uid(self)
        People.created_people += 1
        
        self.epi_survival_rate = 1

        if if_epi:
            if (self.Mother is not None):
                self.epi_survival_rate = 1 - (1 - self.Mother.get_sibling_effect_mortality()) * epi_h
            else:
                self.epi_survival_rate = 1

    def update(self):
        self.survival_rate = self.get_survival_rate()
        self.mating_willingness = self.get_mating_willingness()
        self.marry_willingness = self.get_marry_willingness()
        
    def mutate(self):
        mutant = Allele(effect=-nrand.randint(1,51))
        if get_random() < 0.5:
            self.Paternal_allele = mutant
        else:
            self.Maternal_allele = mutant
            
        self.Menopause_age = 70 + min(self.Paternal_allele.effect, self.Maternal_allele.effect)
        
    def update_N_young_sib_list(self):
        N_young_sib = 0
        for sib in self.sibling_list:
            if (sib.Sex == 0) and (sib.Age < People.Female_age_cutoff):
                N_young_sib += 1
            elif (sib.Sex == 1) and (sib.Age < People.Male_age_cutoff):
                N_young_sib += 1
                
        self.N_young_sib_list.append(N_young_sib)
    
    def get_sibling_effect_mortality(self):
        
        survival_rate_multiplier = np.mean([survival_N_sib(N_young_sib) for N_young_sib in self.N_young_sib_list])
        
        return survival_rate_multiplier
    
    def get_survival_rate(self):
        if self.Sex == 0:
            primary_mortality = Primary_mortality_with_age_female[self.Age]
        else:
            primary_mortality = Primary_mortality_with_age_male[self.Age]
        
        _survival_rate = (1 - primary_mortality)
        
        if if_epi:
            _survival_rate = _survival_rate * self.epi_survival_rate

        if argv['Maternal_effect_mortality']:
            if (self.Mother is None) and (self.Age <= 10):
                _survival_rate = 1 - 10 * (1-_survival_rate)
        
        if argv['Sibling_effect_mortality']:
            survival_rate_multiplier = self.get_sibling_effect_mortality()
            _survival_rate = _survival_rate * survival_rate_multiplier
        
        return max(0,min(1,_survival_rate)) # ensure return 0<=_survival_rate<=1
    
    
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
                if child_age_min < 3:
                    if not ((child_age_min == 2) and (get_random() < 0.5)):
                        _mating_willingness = 0
                        
        else:
            _mating_willingness = Primary_reproduction_rate_with_age_male[self.Age]
            
        return max(0, min(1, _mating_willingness))
    
    def __del__(self):
        #print('__del__')
        People.destructed_people += 1

class Population:
    def __init__(self, if_marriage=False):
        self.N_male = 0
        self.N_female = 0
        self.Male_list = []
        self.Female_list = []
        self.Dead_female_list = []
        self.Current_generation = 0
        self.pop_survival_rate = 1
        self.update()
        self.N_people_died = 0
        self.if_marriage = if_marriage
        #self.allele_dict = init_allele_dict()
        
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
    
    def marry(self): # Assuming polygyny
        Males_to_marry = []
        Females_unmarried = []
        for people in self.Male_list:
            if get_random() < people.marry_willingness:
                Males_to_marry.append(people)
        for people in self.Female_list:
            if (len(people.Partner) == 0) and (get_random() < people.marry_willingness):
                Females_unmarried.append(people)
        nrand.shuffle(Males_to_marry)
        for i in range(min(len(Males_to_marry),len(Females_unmarried))):
            Males_to_marry[i].Partner.append(Females_unmarried[i])
            Females_unmarried[i].Partner.append(Males_to_marry[i])
            
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
            
        for sibling in Female.offspring_list: # Only account for maternal siblings
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

            if not self.if_marriage:     # random mating
                people.Partner = []
                
            people.survival_rate = people.get_survival_rate()
        
        # If there are too many individuals, keep ~1000 of them.
        if self.N_male + self.N_female > 20000000:
            self.pop_survival_rate = 10000/(self.N_male + self.N_female)
        else:
            self.pop_survival_rate = 1
        
        for sex,Pop_list in zip(['Male', 'Female'],[self.Male_list, self.Female_list]):
            for people in Pop_list:
                
                if get_random() < people.survival_rate - (1 - self.pop_survival_rate):
                    # Survived
                    
                    people.Age += 1
                    
                    if (sex == 'Female') and (people.Age == People.Female_age_cutoff) and (people.Mother is not None):
                        people.Mother.N_mature_children += 1
                    elif (sex == 'Male') and (people.Age == People.Male_age_cutoff) and (people.Mother is not None):
                        people.Mother.N_mature_children += 1
                    
                    if sex == 'Male':
                        Male_list_new.append(people)
                    else:
                        Female_list_new.append(people)
                        
                else:
                    # Died
                    # remove the individual from Mother
                    if people.Mother is not None:
                        if sex == 'Male':
                            people.Mother.N_sons -= 1
                        else:
                            people.Mother.N_daughters -= 1
                        people.Mother.offspring_list.remove(people)
                        assert people not in people.Mother.offspring_list
                        
                    # remove the individual from Father
                    if people.Father is not None:
                        if sex == 'Male':
                            people.Father.N_sons -= 1
                        else:
                            people.Father.N_daughters -= 1
                        people.Father.offspring_list.remove(people)
                        assert people not in people.Father.offspring_list
                        
                    # remove the individual from offspring
                    for offspring in people.offspring_list:
                        if sex == 'Male':
                            offspring.Father = None
                        else:
                            #offspring.Mother = None
                            pass
                    
                    # remove the individual from sibling
                    for sibling in people.sibling_list:
                        if sex == 'Male':
                            sibling.N_Brother -= 1
                        else:
                            sibling.N_Sister -= 1
                        sibling.sibling_list.remove(people)
                        assert people not in sibling.sibling_list
                    
                    # remove the individual from partner
                    
                    for partner in people.Partner:
                        partner.Partner.remove(people)
                        assert people not in partner.Partner
                    
                    self.N_people_died += 1
                    
                    if (sex == 'Female') and (people.Age >= 0):
                        people.Partner = []
                        people.sibling_list = []
                        people.Mother = None
                        people.Father = None
                        #people.offspring_list = []
                        self.Dead_female_list.append(people)
                    #print('people died')
                        
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

        
allele_list = []
for effect in [meno_age - 70, 0]:
    allele_list.append(Allele(effect=effect))


Allele.N = 0
People.created_people = 0
People.destructed_people = 0
#default_allele = Allele(effect=0)
#default_allele = allele_list[0]
default_allele = allele_list[0]

Pop = Population()
for sex in [0,1]:
    for age in range(10):
        for i in range(200):
            people = People(sex=sex,Paternal_allele=default_allele,Maternal_allele=default_allele,gen_of_birth=-age,age=age)
            Pop.Add_people(people)

people=None
allele_list_dict={}
N_list = []
sex_ratio_list = []
sex_ratio_at_birth_list = []

N_birth_list = []
N_children_list = []
N_birth_WT_list = []
N_children_WT_list = []
N_birth_MT_list = []
N_children_MT_list = []

Menopause_age_list = []

for ii in range(200):
    print(f'{ii}   ',end='\r')
    Pop.reproduce()
    Pop.next_generation()
    N_list.append(Pop.N_male+Pop.N_female)
    #allele_dict = Pop.get_allele_dict()

    if Pop.N_female + Pop.N_male > 100000:
        break

    #for key,value in allele_dict.items():
    #    if key in allele_list_dict:
    #        allele_list_dict[key].append(value)
    #    else:
    #        allele_list_dict[key] = [value]

    #Menopause_age_list.append(Pop.get_mean_Menopause_age())


N_birth_list = []
N_mature_children_list = []
for people in Pop.Dead_female_list:
    if (people.Age >= People.Female_age_cutoff) and (people.Gen_of_birth < Pop.Current_generation - 100) and (people.Gen_of_birth >= 0):
        N_birth_list.append(people.N_birth)
        N_mature_children_list.append(people.N_mature_children)

N_birth_list = np.array(N_birth_list)
N_mature_children_list = np.array(N_mature_children_list)

NMC_mean = N_mature_children_list.mean()
NMC_se = N_mature_children_list.std()/np.sqrt(len(N_mature_children_list))

#with open(f'./individual_level_trace_mother_menopause_age_results_2/{meno_age}.txt', 'w') as f:
#    f.write(f'{meno_age}\t{NMC_mean}\t{NMC_se}\n')

with open(f'./individual_level_trace_mother_menopause_age_results_5/{meno_age}.txt', 'w') as f:
    for line in N_list:
        f.write(f"{line}\n")
