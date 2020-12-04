import csv
from lxml import etree as et
from argparse import ArgumentParser
from collections import defaultdict
import sys
from copy import deepcopy
import numpy as np

# load BEAST xml
def load_xml(xmlfile):
    parser = et.XMLParser(remove_blank_text=True)
    tree = et.parse(xmlfile,parser=parser)
    return tree

# load travel history csv
def load_hist(c):
    columns = defaultdict(list)
    with open(c) as f:
        reader = csv.DictReader(f)
        assert set(['name','travelHistory','travelDays','priorMean','priorStdev']).issubset(set(reader.fieldnames)),\
                "Cannot find column names in csv file, please check documentation for appropriate formatting"
        for row in reader:
            for (col,value) in row.items():
                columns[col].append(value)
    return columns

# check that all sequences in metadata file are in the xml
def check_sequences(xml,hist):
    trav_names = hist['name']
    taxa_names = get_all_taxa(xml)
    return set(trav_names).issubset(set(taxa_names))

# all taxa ids from xml
def get_all_taxa(xml):
    return [x.attrib['id'] for x in xml.getroot().find('taxa').findall("taxon")]

# get name of discrete trait
def get_trait_name(xml):
    return xml.getroot().find("attributePatterns").attrib['attribute']

# get all unique locations from xml
def get_all_locations(xml):
    return [x.attrib['code'] for x in xml.getroot().find('generalDataType').findall("state")]

# get all ambiguity codes from xml
def get_all_ambiguities(xml):
    return [x.attrib['code'] for x in xml.getroot().find('generalDataType').findall("ambiguity")]

# get all travel history origin locations from csv file
def get_travel_locations(hist):
    return list(set(hist['travelHistory']))

# make taxa element for ancestral taxa
def make_ancestral_taxa(xml,hist):
    names = hist['name']
    anc_locs = hist['travelHistory']
    ancestral_taxa = et.Element('taxa',id='ancestralTaxa')
    for name,loc in zip(names,anc_locs):
        attr = et.Element('attr',name='location')
        attr.text = loc
        taxon = et.Element("taxon",id=name +'_ancestor_taxon')
        taxon.append(attr)
        ancestral_taxa.append(taxon)
    return ancestral_taxa

# make taxa element combining taxa and ancestral taxa
def make_all_taxa():
    alltaxa = et.Element('taxa',id='allTaxa')
    alltaxa.append(et.Element('taxa',idref='taxa'))
    alltaxa.append(et.Element('taxa',idref='ancestralTaxa'))
    return alltaxa

def add_new_taxa_elements(xml,hist):
    root = xml.getroot()
    anc_taxa = make_ancestral_taxa(xml,hist)
    alltaxa = make_all_taxa()
    # add Ancestral Taxa element to xml tree
    root.insert(root.index(root.find("taxa"))+1,anc_taxa)
    # add allTaxa element to xml tree
    root.insert(root.index(root.find("taxa"))+2,alltaxa)
    # change attribute patterns reference to allTaxa
    root.find('attributePatterns').find("taxa").attrib['idref'] = "allTaxa"
    return


# make xml elements for the new locations to add to generalDataType
def make_new_locations(xml,hist):
    new_location_states = []
    taxa_locs = get_all_locations(xml)
    taxa_amb = get_all_ambiguities(xml)
    trav_locs = get_travel_locations(hist)
    new_locs = [x for x in trav_locs if (("_" not in x) and (x not in taxa_locs))]
    new_ambiguity_codes = [x for x in trav_locs if (("_" in x) and (x not in taxa_amb))]
    for n in new_ambiguity_codes:
        for l in n.split("_"):
            if (l not in taxa_locs) and (l not in new_locs):
                new_locs.append(l)
    for x in new_locs:
        new_location_states.append(et.Element('state', code=x))
    for x in new_ambiguity_codes:
        new_location_states.append(et.Element('ambiguity', code=x, states=" ".join(x.split("_"))))
    return new_location_states

# add new locations and ambiguity codes to generalDataType
def add_new_locs(xml,hist):
    new_locs = make_new_locations(xml,hist)
    gen_dt = xml.getroot().find("generalDataType")
    old_locs = []
    for i in gen_dt.findall("state"):
        # make a copy of the element and remove
        old_locs.append(deepcopy(i))
        gen_dt.remove(i)
    all_locs = sorted(old_locs + new_locs,key=lambda x:x.attrib['code'])
    for l in all_locs:
        gen_dt.append(l)
    return

def parse_covariate(covname):
    rows = []
    with open(covname) as f:
        for line in f:
            rows.append(line.strip().split(','))

    mat = np.array(rows)
    values = mat[1:,1:]
    try:
        values = values.astype(float)
    except:
        "Matrix contains non-numeric entries"
    cov_name = mat[0,0]
    assert list(mat[1:,0])==list(mat[0,1:]),f'Column names do not match row names in covariate {cov_name}'
    state_names = list(mat[1:,0])
    assert values.shape[0] == values.shape[1], f'Covariate {cov_name} matrix is not square'
    return cov_name,state_names,values

def transform_matrix(name,values):
    num_states = values.shape[0]
    upper_tri_indices = np.triu_indices(num_states,1)
    upper_tri = values[upper_tri_indices]
    lower_tri = values.T[upper_tri_indices]
    vector_mat = np.concatenate([upper_tri,lower_tri])
    if 0 in vector_mat:
        x = input(f"The matrix '{name}' contains 0 entries, covariate will not be transformed, press y to continue: ")
        if x in ['y','Y']:
            return vector_mat
    else:
        log_mat = np.log(vector_mat)
        mean = np.mean(log_mat)
        std = np.std(log_mat)
        return (log_mat - mean)/std

# edit number of rates to correspond with new generalDatatype
def edit_number_of_rates(xml,glm_covariates = None):
    dta_model = check_DTA(xml)
    root = xml.getroot()
    trait_name = get_trait_name(xml)
    num_locs = len([x.attrib['code'] for x in root.find('generalDataType').findall("state")])
    if dta_model == 'glm':
        (root.find("glmSubstitutionModel")
            .find('rootFrequencies')
            .find('frequencyModel')
            .find('frequencies')
            .find("parameter")).attrib['dimension'] = str(int(num_locs))
        # update covariate vector values
        if glm_covariates != None:
            for cov in glm_covariates:
                cov_name,state_names,matrix = parse_covariate(cov)
                transformed = transform_matrix(cov_name,matrix)
                transformed = " ".join(transformed.astype(str))
                xml_covs = (root.find("glmSubstitutionModel")
                    .find('glmModel')
                    .find('independentVariables')
                    .find('designMatrix').findall("parameter"))
                cov_param = [x for x in xml_covs if x.attrib['id']==f"{trait_name}.{cov_name}"]
                assert len(cov_param) == 1, f"Covariate name '{cov_name}' not found in xml, make sure names match"
                assert matrix.shape[0]==num_locs,\
                    f"Covariate '{cov_name}' matrix dimensions {matrix.shape} don't match the augmented number of states ({num_locs}), was the covariate matrix augmented? \nxml_states\n{get_all_locations(xml)}\nmatrix\n{state_names}"
                assert state_names == get_all_locations(xml), \
                    f"Unable to match column names in covariate matrix with trait states, make sure it is alphabetically ordered\n{state_names}\n{get_all_locations(xml)}"

                cov_param[0].attrib['value'] = transformed

    else:
        if dta_model == 'symmetric':
            num_rates = num_locs*(num_locs-1)/2
        elif dta_model == 'asymmetric':
            num_rates = num_locs*(num_locs-1)
        (root.find("generalSubstitutionModel")
            .find('frequencies')
            .find('frequencyModel')
            .find('frequencies')
            .find("parameter")).attrib['dimension'] = str(int(num_locs))
        (root.find("generalSubstitutionModel")
            .find('rates')
            .find('parameter')).attrib['dimension'] = str(int(num_rates))
        (root.find("generalSubstitutionModel")
            .find('rateIndicator')
            .find('parameter')).attrib['dimension'] = str(int(num_rates))
    return

def edit_markov_jumps_tree_likelihood(xml):
    check_mj(xml)
    mjumps = xml.getroot().find("markovJumpsTreeLikelihood")
    # replace treemodel with ancestral trait treemodel
    tm = mjumps.find("treeModel")
    mjumps.insert(mjumps.index(tm)+1,et.Element("ancestralTraitTreeModel",
                                                idref='ancestralTraitTreeModel'))
    mjumps.remove(tm)
    # remove mj counts
    counts = [x for x in mjumps.findall("parameter") if "count" in x.attrib['id']][0]
    mjumps.remove(counts)

    if check_DTA(xml) == 'asymmetric':
        loc_root_freq = mjumps.find("frequencyModel").find('frequencies').find("parameter")
        loc_root_freq.attrib['dimension'] = str(len(get_all_locations(xml)))
    return

def check_DTA(xml):
    xmlstr = et.tostring(xml).decode()
    if "-- symmetric CTMC" in xmlstr:
        return 'symmetric'
    elif "-- asymmetric CTMC" in xmlstr:
        return 'asymmetric'
    elif "-- GLM substitution" in xmlstr:
        return 'glm'

def check_mj(xml):
    assert xml.getroot().findall("markovJumpsTreeLikelihood") != [],\
        "markovJumpsTreeLikelihood element not found, make sure you are reconstructing complete change history on tree"
    return True



def make_ancestral_trait_elements(xml,hist):
    new_ops = []
    new_priors = []
    new_parameters_log = []
    ancestralTree = et.Element('ancestralTraitTreeModel',id='ancestralTraitTreeModel')
    ancestralTree.append(et.Element('treeModel',idref='treeModel'))

    for i,(name,days,pr_mean,pr_stdev) in enumerate(zip(hist['name'],hist['travelDays'],hist['priorMean'],hist['priorSTDEV'])):
        anc_name = name + "_ancestor_taxon"
        # create ancestor element
        anc = et.Element("ancestor")
        anc.append(et.Element("taxon", idref=anc_name))
        anc.append(et.Element("parameter", id=f"pseudoBranchLength{i+1}",
                                 value="0.000", lower='0.0'))
        # ancestral path element
        ancestralPath = et.Element("ancestralPath",relativeToTipHeight='true')
        ancestralPath.append(et.Element("taxon", idref=name))
        # add travel dates
        try:
            days = int(days)
            ancestralPath.append(et.Element("parameter", id=f"time{i+1}",lower='0.0',
                                              value=f"{round(days/365,10)}"))
        # if no travel dates are available sample from Normal prior
        except:
            # default prior normal 10 days 3 days stdev
            pr_mean = int(pr_mean)
            pr_stdev = int(pr_stdev)
            ancestralPath.append(et.Element("parameter", id=f"time{i+1}",lower='0.0',
                                              value=f"{round(pr_mean/365,10)}"))
            # create corresponding prior and operator elements
            normalPrior = et.Element("normalPrior",mean=f"{round(int(pr_mean)/365,10)}",stdev=f"{round(pr_stdev/365,10)}")
            normalPrior.append(et.Element("parameter",idref=f"time{i+1}"))
            new_priors.append(normalPrior)
            operator =  et.Element("scaleOPerator",scaleFactor='0.75',weight='0.05')
            operator.append(et.Element("parameter",idref=f"time{i+1}"))
            new_ops.append(operator)
            new_parameters_log.append(et.Element("parameter",idref=f"time{i+1}"))
        anc.append(ancestralPath)
        ancestralTree.append(anc)

    return ancestralTree,new_ops,new_priors,new_parameters_log

def add_ancestral_trait_elements(xml,hist):
    ancestralTree,new_ops,new_priors,new_parameters_log = make_ancestral_trait_elements(xml,hist)
    root = xml.getroot()
    # add ancestral tree model
    root.insert(root.index(root.find('treeDataLikelihood'))+1,ancestralTree)
    # add new priors
    priors = root.find("mcmc").find("joint").find("prior")
    for p in new_priors:
        priors.append(p)
    # add new operators
    operators = root.find("operators")
    for o in new_ops:
        operators.append(o)
    # add new parameters to log
    filelog = [x for x in root.find("mcmc").findall("log") if x.attrib['id']=='fileLog'][0]
    for p in new_parameters_log:
        filelog.append(p)
    return

def edit_tree_logs(xml):
    trait_name = get_trait_name(xml)
    # edit markov jump history tree logs
    try:
        history_trees = [x for x in xml.getroot().find("mcmc").findall("logTree") if ("history" in x.attrib['fileName'])][0]
    except:
        print("Can't find tree Markov jump history logTree, make sure you added in Beauti")
        return

    # add trait annotation to tree history log
    loc_states = et.Element("trait",name=f'{trait_name}.states', tag=f'{trait_name}')
    loc_states.append(et.Element('markovJumpsTreeLikelihood',idref=f"{trait_name}.treeLikelihood"))
    history_trees.append(loc_states)
    #replace treemodel with ancestralTraitTreeModel
    tmodel = history_trees.find("treeModel")
    history_trees.remove(tmodel)
    history_trees.append(et.Element("ancestralTraitTreeModel",idref="ancestralTraitTreeModel"))
    # edit regular tree log file
    trees = [x for x in xml.getroot().find("mcmc").findall("logTree") \
                       if "history" not in x.attrib['fileName']][0]
    # remove markov jumps counts from tree
    empirical_trees = [x for x in xml.getroot().find("mcmc").findall("logTree") \
                       if "history" not in x.attrib['fileName']][0]
    counts = [x for x in trees.findall("trait") if "count" in x.attrib['name']][0]
    empirical_trees.remove(counts)
    #replace treemodel with ancestralTraitTreeModel
    tmodel2 = empirical_trees.find("treeModel")
    empirical_trees.remove(tmodel2)
    empirical_trees.append(et.Element("ancestralTraitTreeModel",idref="ancestralTraitTreeModel"))
    # replace ancestralTreeLikelihood with markovJumpsTreeLikelihood for discrete states
    traits = [x for x in empirical_trees.findall("trait")]
    loctrait = [x for x in traits if (x.attrib['name'] == f"{trait_name}.states")][0]
    anc_tl = loctrait.find("ancestralTreeLikelihood")
    loctrait.remove(anc_tl)
    loctrait.append(et.Element("markovJumpsTreeLikelihood",idref=f"{trait_name}.treeLikelihood"))
    # add treeModel treeLog
    treeModel_trees =et.Element("logTree",
                        logEvery=f"{empirical_trees.attrib['logEvery']}",
                        nexusFormat="true",
                        fileName=f"noanc_{empirical_trees.attrib['fileName']}",
                        sortTranslationTable="true")
    treeModel_trees.append(et.Element("treeModel",idref='treeModel'))
    for i in empirical_trees.findall("trait"):
        if i.attrib['name'] == 'rate':
            treeModel_trees.append(deepcopy(i))
    treeModel_trees.append(et.Element("joint",idref='joint'))

    xml.getroot().find("mcmc").append(et.Comment(" write non-augmented trees "))
    xml.getroot().find("mcmc").append(treeModel_trees)

    return

def write_xml(xml,filename):
    xml_str = et.tostring(xml,pretty_print=True).decode()
    xml_str = xml_str.replace("<!",'\n<!').replace("->",'->\n')
    if filename!=None:
        with open(filename,'w') as f:
            f.write(xml_str)
    else:
        print(xml_str)
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--xml",required=True,help='BEAST XML file')
    parser.add_argument("--hist",required=True,help='CSV file containing the travel history metadata')
    parser.add_argument("--out",default=None,help='output file name')
    parser.add_argument("--covariate",default=None,action='append',help='updated covariate matrix')
    args=parser.parse_args()

    trav_hist = load_hist(args.hist)
    xml = load_xml(args.xml)
    assert check_sequences(xml,trav_hist), "All names in travel history not found in xml file"
    add_new_taxa_elements(xml,trav_hist)
    add_new_locs(xml,trav_hist)
    edit_number_of_rates(xml,glm_covariates=args.covariate)
    add_ancestral_trait_elements(xml,trav_hist)
    edit_markov_jumps_tree_likelihood(xml)
    edit_tree_logs(xml)
    write_xml(xml,args.out)
