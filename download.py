#!/usr/bin/env python

import argparse
import os
import requests
import json

from retrying import retry


ASSET_URL = 'https://api.planet.com/data/v1/item-types/{}/items/{}/assets/'
SEARCH_URL = 'https://api.planet.com/data/v1/quick-search'

# set up auth
SESSION = requests.Session()
SESSION.auth = ("3d65c0d0a23144e0b56aade1b66ef028", '')


class RateLimitException(Exception):
    pass


def handle_page(page):
    return [item['id'] for item in page['features']]


def retry_if_rate_limit_error(exception):
    """Return True if we should retry (in this case when it's a rate_limit
    error), False otherwise"""
    return isinstance(exception, RateLimitException)


def check_status(result, msg=None):
    if result.status_code == 429:
        print 'Rate limit error - retrying'
        raise RateLimitException('rate limit error')
    else:
        if msg:
            print msg
        return True


@retry(
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    retry_on_exception=retry_if_rate_limit_error,
    stop_max_attempt_number=5)
def run_search(search_request):
    print 'Running query'

    result = SESSION.post(SEARCH_URL, json=search_request)

    check_status(result)

    page = result.json()
    final_list = handle_page(page)

    while page['_links'].get('_next') is not None:
        page_url = page['_links'].get('_next')
        page = SESSION.get(page_url).json()
        ids = handle_page(page)
        final_list += ids

    return [fid for fid in final_list]


@retry(
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    retry_on_exception=retry_if_rate_limit_error,
    stop_max_attempt_number=5)
def activate(item_id, item_type, asset_type):
    result = SESSION.get(ASSET_URL.format(item_type, item_id))

    check_status(result)

    status = result.json()[asset_type]['status']
    if status == 'active':
        print 'Item already active: {}'.format(item_id)
        return False
    else:
        item_activation_url = result.json()[asset_type]['_links']['activate']

        print 'Activating {} {} for {}'.format(item_type, asset_type, item_id)
        result = SESSION.post(item_activation_url)

        return check_status(result, 'Activation process started successfully')


@retry(
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    retry_on_exception=retry_if_rate_limit_error,
    stop_max_attempt_number=5)
def check_activation(item_id, item_type, asset_type):
    result = SESSION.get(ASSET_URL.format(item_type, item_id))

    check_status(result)

    status = result.json()[asset_type]['status']
    print '{}: {}'.format(item_id, status)

    if status == 'active':
        return True
    else:
        print 'Item not yet active: {}'.format(item_id)
        return False


@retry(
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    retry_on_exception=retry_if_rate_limit_error,
    stop_max_attempt_number=5)
def download(url, path, item_id, asset_type, overwrite):
    fname = '{}_{}.tif'.format(item_id, asset_type)
    local_path = os.path.join(path, fname)

    if not overwrite and os.path.exists(local_path):
        print 'File {} exists - skipping ...'.format(local_path)
    else:
        print 'Downloading file to {}'.format(local_path)
        # memory-efficient download, per
        # stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
        result = requests.get(url)

        if check_status(result):
            f = open(local_path, 'wb')
            for chunk in result.iter_content(chunk_size=512 * 1024):
                # filter out keep-alive new chunks
                if chunk:
                    f.write(chunk)
            f.close()

    return True


def process_activation(func, id_list, item_type, asset_type):
    results = []

    for item_id in id_list:
        result = func(item_id, item_type, asset_type)
        results.append(result)

    return results


def process_download(path, id_list, item_type, asset_type, overwrite):
    results = []

    # ensure directory structure exists
    try:
        os.makedirs(path)
    except OSError:
        pass

    # now start downloading each file
    for item_id in id_list:
        result = SESSION.get(ASSET_URL.format(item_type, item_id))

        if result.json()[asset_type]['status'] == 'active':
            download_url = result.json()[asset_type]['location']
            result = download(download_url, path, item_id, asset_type, overwrite)
        else:
            result = False

        results.append(result)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idlist', help='Location of file containing image ids (one per line) to process')
    parser.add_argument('--query', help='Path to json file containing query')
    parser.add_argument('--search', help='Search for images', action='store_true')
    parser.add_argument('--activate', help='Activate assets', action='store_true')
    parser.add_argument('--check', help='Check activation status', action='store_true')
    parser.add_argument('--download', help='Path where downloaded files should be stored')
    parser.add_argument('--overwrite', help='Overwrite existing downloads', action='store_true')

    parser.add_argument('item', help='Item type (e.g. REOrthoTile or PSOrthoTile)')
    parser.add_argument('asset', help='Asset type (e.g. visual, analytic, analytic_xml)')

    args = parser.parse_args()

    if not (args.idlist or args.query):
        parser.error('Error: please supply an --idlist or --query argument.')

    if args.idlist:
        with open(args.idlist) as f:
            id_list = [i.strip() for i in f.readlines()]

    if args.query:
        # load query json file
        with open(args.query, 'r') as fp:
            query = json.load(fp)

        # Search API request object
        search_payload = {'item_types': [args.item], 'filter': query}

        id_list = run_search(search_payload)

    if args.search:
        print '%d available images' % len(id_list)

    elif args.activate:
        results = process_activation(activate, id_list, args.item, args.asset)
        msg = 'Requested activation for {} of {} items'
        print msg.format(results.count(True), len(results))

    # check activation status of all data returned by search query
    elif args.check:
        results = process_activation(check_activation, id_list, args.item,
                                     args.asset)

        msg = '{} of {} items are active'
        print msg.format(results.count(True), len(results))

    # download all data returned by search query
    elif args.download:
        results = process_download(args.download, id_list, args.item,
                                   args.asset, args.overwrite)
        msg = 'Successfully downloaded {} of {} files to {}. {} were not activated yet.'
        print msg.format(results.count(True), len(results), args.download, results.count(False))

    else:
        parser.error('Error: no action supplied. Please check help (--help) or revise command.')


'''Sample commands, for testing.
python download.py --query redding.json --search PSScene3Band visual
python download.py --query redding.json --check PSScene3Band visual
python download.py --query redding.json --activate PSScene3Band visual
python download.py --query redding.json --download /tmp PSScene3Band visual
python download.py --idlist ids_small.txt --check PSScene3Band visual
python download.py --idlist ids_small.txt --activate PSScene3Band visual
python download.py --idlist ids_small.txt --download /tmp PSScene3Band visual
'''
