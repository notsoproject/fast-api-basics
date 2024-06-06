import string
import pandas as pd
import numpy as np
import time
import os
import gc
import Levenshtein
from Levenshtein import ratio


def preprocess_text(text):
    """
    Preprocesses text by converting to lowercase, and removing punctuation and spaces efficiently.

    Args:
        text: The text to preprocess.

    Returns:
        The preprocessed text as a single string.
    """
    # Ensure the input is a string
    if not isinstance(text, str):
        text = str(text)
    
    # Define punctuation and space characters to remove
    remove_chars = string.punctuation + ' '
    
    # Create translation table: maps each punctuation and space to None (i.e., it will be removed)
    table = str.maketrans('', '', remove_chars)

    # Apply translation table, convert to lowercase in one go
    preprocessed_text = text.translate(table).lower()

    return preprocessed_text



#---------------------------Defining Class---------
class TrieNode:
    def __init__(self):
        self.children = {}  # Add this line to `TrieNode`
        self.is_end_of_word = False
        self.data = {'FName': None,'MiddleName': None,'LastName': None, 'FatherName': None, 'IDNo': None,'CIFID': None}
        self.has_data = True  # Flag to indicate node stores data (optional)

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, Fname,Mname,Lname, father_name, idno, cifid):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.data['FName'] = Fname
        node.data['MiddleName'] = Mname
        node.data['LastName'] = Lname
        node.data['FatherName'] = father_name
        node.data['IDNo'] = idno
        node.data['CIFID'] = cifid

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False, None, None
            node = node.children[char]
        return node.is_end_of_word, node.data.get('FName'),node.data.get('MiddleName'),node.data.get('LastName'), node.data.get('FatherName'), node.data.get('IDNo'),node.data.get('CIFID')

    def search_fuzzy(self, query, threshold=0.85): #adjust threshold here as necessary
        results = []
        self._search_fuzzy_recursive(query, self.root, '', threshold, results)
        return results

    def _search_fuzzy_recursive(self, query, node, prefix, threshold, results):
        # Check base cases:
        if not node:  # Reached an empty node (no matching prefix so far)
            return

        # Explore child nodes for potential insertions
        for child, child_node in node.children.items():
            new_prefix = prefix + child
            self._search_fuzzy_recursive(query, child_node, new_prefix, threshold, results)

        # Check similarity and add exact matches with data (if applicable)
        similarity = Levenshtein.ratio(query, prefix)
        if similarity >= threshold and node.is_end_of_word and (hasattr(node, 'has_data') and node.has_data):  # Check for data flag (optional)
            results.append((prefix, node.data))

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


#--------BATCH PROCESSING------------


def insert_data(trie, df):
    for index, row in df.iterrows():
        combined_text = row['Combined']
        preprocessed_word = preprocess_text(combined_text)
        trie.insert(
            word=preprocessed_word, 
            Fname=row['FName'],
            Mname=row['MiddleName'],
            Lname=row['LastName'], 
            father_name=row['FatherName'], 
            idno=row['IDNo'],
            cifid=row['CIFID']
        )
        

def search_in_trie(trie, query):
    # Preprocess the text
    query = preprocess_text(query)
    results = trie.search_fuzzy(query)
    detailed_results = []

    if results:
        for prefix, data in results:
            match_detail = {
                "prefix": prefix,
                "FName": data.get('FName'),
                "LastName": data.get('LastName'),
                "MiddleName": data.get('MiddleName'),
                "FatherName": data.get('FatherName'),
                "IDNo": data.get('IDNo'),
                "CIFID": data.get('CIFID')
            }
            detailed_results.append(match_detail)
            # Print for debugging purposes, can be removed or commented out in production
            print(f"- {prefix} \n FName: {data.get('FName')} \n LastName: {data.get('LastName')} \n MiddleName: {data.get('MiddleName')} \n  Father Name: {data.get('FatherName')}\n Idno: {data.get('IDNo')}\n CIFID: {data.get('CIFID')}")

    return detailed_results


def process_batches(queries, df, batch_size=750000):  #optimal batch size after preliminary assessment
    print(f"Total queries: {len(queries)}")
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    results = {query: [] for query in queries}  # Initialize results with empty lists for each query
    
    for batch_num in range(num_batches):
        print(f"Processing batch {batch_num+1}/{num_batches}...")
        batch_start = batch_num * batch_size
        batch_end = batch_start + batch_size
        batch_df = df.iloc[batch_start:batch_end]

        trie = Trie()
        insert_data(trie, batch_df)  # Assuming insert_data populates the trie with the batch data

        for query in queries:
            matches = search_in_trie(trie, query)  # search_in_trie now returns list of match details
            if matches:
                print(f"Found {len(matches)} matches for query '{query}' in batch number {batch_num+1}")
                results[query].extend(matches)  # Appending all matches found for this query in this batch

        # Clean up
        del trie
        gc.collect()  # Explicit garbage collection

    return results



#-----------------STORING OUTPUT IN CSV----------

def process_results(results, df_test):
    keys = []
    prefixes = []
    first_names = []
    last_names = []
    middle_names = []
    father_names = []
    id_numbers = []
    CIFID = []
    blacklist_numbers = []

    for key, values in results.items():
        # Retrieve the BlacklistNumber for the current key
        blacklist_number = df_test.loc[df_test['Combined'] == key, 'BlacklistNumber'].values[0] if key in df_test['Combined'].values else None
        
        if not values:
            # If the value is an empty list, add a row with null values but include the BlacklistNumber
            keys.append(key)
            prefixes.append(None)
            first_names.append(None)
            last_names.append(None)
            middle_names.append(None)
            father_names.append(None)
            id_numbers.append(None)
            CIFID.append(None)
            blacklist_numbers.append(blacklist_number)  # Always append BlacklistNumber
        else:
            for value in values:
                # Extract details from each entry
                prefix = value.get('prefix')
                first_name = value.get('FName')
                last_name = value.get('LastName')
                middle_name = value.get('MiddleName')
                father_name = value.get('FatherName')
                id_number = value.get('IDNo')
                cif_ids = value.get('CIFID')

                # Append parsed data to lists
                keys.append(key)
                prefixes.append(prefix)
                first_names.append(first_name)
                last_names.append(last_name)
                middle_names.append(middle_name)
                father_names.append(father_name)
                id_numbers.append(id_number)
                CIFID.append(cif_ids)
                blacklist_numbers.append(blacklist_number)  # Always append BlacklistNumber

    # Create DataFrame
    df_output = pd.DataFrame({
        'Key': keys,
        'Prefix': prefixes,
        'First Name': first_names,
        'Last Name': last_names,
        'Middle Name': middle_names,
        'Father Name': father_names,
        'ID Number': id_numbers,
        'CIFID': CIFID,
        'Blacklist Number': blacklist_numbers  # Add BlacklistNumber column
    })

    # Concatenate first name, middle name, and last name with white space separator
    df_output['Full Name'] = df_output[['First Name', 'Middle Name', 'Last Name']].apply(lambda x: ' '.join(x.dropna()), axis=1)

    # Drop individual name columns if desired
    df_output.drop(['First Name', 'Middle Name', 'Last Name', 'Prefix'], axis=1, inplace=True)

    return df_output
