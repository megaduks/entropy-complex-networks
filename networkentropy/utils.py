from bs4 import BeautifulSoup
import requests
import wget
import tarfile
import os
import shutil

from builtins import object
from typing import Dict, List
import networkx as nx


def read_avalilable_datasets_konect() -> List[str] :
    """
    Reads the list of all networks available through the Koblenz network repository
    :return: list of network names
    """

    base_url = "http://konect.uni-koblenz.de/downloads/"
    response = requests.get(base_url)

    if response.status_code != 200:
        print("An error occurred while getting data.")
    else:
        html = response.content
        soup = BeautifulSoup(html, "html5lib")

        table_html = soup.find(id='sort1')
        tbody_html = table_html.find('tbody')
        rows = tbody_html.findAll('tr')

        values = [
            [
                cell.get('href')
                for cell
                in value('a')
                if 'tsv' in cell.get('href')
            ]
            for value
            in rows
        ]

        return [
            val[0].replace('.tar.bz2', '').replace('tsv/', '')
            for val
            in values
        ]


def download_tsv_dataset_konect(network_name: str, dir_name: str) -> str:
    """
    Downloads the compressed network file into local directory

    :param network_name: name of the network to download
    :param dir_name: name of the local directory to which the compressed file should be saved
    :return: name of the downloaded file
    """

    assert (network_name in read_avalilable_datasets_konect()), \
        "No network named: '" + network_name + "' found in Konect!"

    tsv_file = 'http://konect.uni-koblenz.de/downloads/tsv/' + network_name + '.tar.bz2'
    output_file = network_name + '.tar.bz2'
    file_name = wget.download(tsv_file, out=output_file)

    if os.path.exists(output_file):
        shutil.move(file_name, dir_name + output_file)

    return output_file


def unpack_tar_bz2_file(file_name: str, dir_name: str) -> str:
    """
    Unpacks the downloaded compressed file on disk

    :param file_name: name of the compressed file
    :param dir_name: name of the directory in which unpacking happens
    :return: name of the directory where unpacked files are
    """

    tar = tarfile.open(dir_name + file_name, "r:bz2")
    output_dir = dir_name + "network_" + file_name.replace('.tar.bz2', '') + "/"

    tar.extractall(output_dir)
    tar.close()

    return output_dir + file_name.replace('.tar.bz2', '/')


def build_network_from_out_konect(network_name: str, dir_name: str) -> nx.Graph:
    """
    Reads network files stored on disk and builds a proper NetworkX graph object

    :param network_name: name of the network to build
    :param dir_name: name of the directory to download files to
    :return: NetworkX graph object
    """

    file_name = download_tsv_dataset_konect(network_name=network_name, dir_name=dir_name)
    output_dir = unpack_tar_bz2_file(file_name=file_name, dir_name=dir_name)

    files = [
        file
        for file
        in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, file))
    ]

    out_file = [
        file
        for file
        in files
        if 'out.' in file
    ]

    assert (len(out_file) > 0), 'No out. file in the directory.'

    G = nx.read_adjlist(output_dir + out_file[0], comments='%')

    return G
