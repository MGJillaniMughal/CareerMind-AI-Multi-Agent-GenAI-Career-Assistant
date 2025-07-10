"""
CareerMind AI - Multi-Agent GenAI Career Assistant
Enhanced Job Search System with LinkedIn Integration and Advanced Features
"""

import aiohttp
import os
import urllib
import asyncio
import requests
import logging
from typing import List, Literal, Union, Optional, Dict, Any
from asgiref.sync import sync_to_async
from bs4 import BeautifulSoup
from datetime import datetime
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Employment type mappings for LinkedIn API
employment_type_mapping = {
    "full-time": "F",
    "contract": "C", 
    "part-time": "P",
    "temporary": "T",
    "internship": "I",
    "volunteer": "V",
    "other": "O",
}

experience_type_mapping = {
    "internship": "1",
    "entry-level": "2", 
    "associate": "3",
    "mid-senior-level": "4",
    "director": "5",
    "executive": "6",
}

job_type_mapping = {
    "onsite": "1",
    "remote": "2", 
    "hybrid": "3",
}


def build_linkedin_job_url(
    keywords: str,
    location: str = None,
    employment_type: List[str] = None,
    experience_level: List[str] = None,
    job_type: List[str] = None,
    start: int = 0,
) -> str:
    """
    Build enhanced LinkedIn job search URL with comprehensive filtering.
    
    Args:
        keywords: Search keywords
        location: Geographic location
        employment_type: List of employment types
        experience_level: List of experience levels
        job_type: List of job types (remote/onsite/hybrid)
        start: Starting position for pagination
        
    Returns:
        str: Complete LinkedIn job search URL
    """
    try:
        base_url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search/"
        
        # Prepare query parameters
        query_params = {
            "keywords": keywords,
            "start": start,
            "sortBy": "R"  # Relevance sorting
        }

        if location:
            query_params["location"] = location

        # Handle employment type filtering
        if employment_type:
            if isinstance(employment_type, str):
                employment_type = [employment_type]
            mapped_types = [employment_type_mapping.get(et, et) for et in employment_type]
            query_params["f_WT"] = ",".join(mapped_types)

        # Handle experience level filtering
        if experience_level:
            if isinstance(experience_level, str):
                experience_level = [experience_level]
            mapped_levels = [experience_type_mapping.get(el, el) for el in experience_level]
            query_params["f_E"] = ",".join(mapped_levels)

        # Handle job type filtering (remote/onsite/hybrid)
        if job_type:
            if isinstance(job_type, str):
                job_type = [job_type]
            mapped_job_types = [job_type_mapping.get(jt, jt) for jt in job_type]
            query_params["f_CF"] = ",".join(mapped_job_types)

        # Build the complete URL
        query_string = urllib.parse.urlencode(query_params)
        full_url = f"{base_url}?{query_string}"
        
        logger.info(f"Built LinkedIn URL for keywords: '{keywords}'")
        return full_url
        
    except Exception as e:
        logger.error(f"Error building LinkedIn URL: {str(e)}")
        raise


def validate_job_search_params(agent_input: Union[str, list], value_dict_mapping: dict) -> Union[str, list, None]:
    """
    Validate and clean job search parameters.
    
    Args:
        agent_input: Input parameter to validate
        value_dict_mapping: Mapping dictionary for validation
        
    Returns:
        Validated parameter or None if invalid
    """
    try:
        if isinstance(agent_input, list):
            validated_list = []
            for input_str in agent_input:
                if value_dict_mapping.get(input_str):
                    validated_list.append(input_str)
            return validated_list if validated_list else None
            
        elif isinstance(agent_input, str):
            return agent_input if value_dict_mapping.get(agent_input) else None
        
        return None
        
    except Exception as e:
        logger.error(f"Error validating job search params: {str(e)}")
        return None


def get_job_ids_from_linkedin_api(
    keywords: str,
    location_name: str = None,
    employment_type: List[str] = None,
    limit: Optional[int] = 10,
    job_type: List[str] = None,
    experience: List[str] = None,
    listed_at: int = 86400,
    distance: int = None,
) -> List[str]:
    """
    Get job IDs using LinkedIn API (when available).
    
    Args:
        keywords: Search keywords
        location_name: Geographic location
        employment_type: Employment type filters
        limit: Maximum number of jobs
        job_type: Job type filters
        experience: Experience level filters
        listed_at: Time since posting (seconds)
        distance: Search radius
        
    Returns:
        List of job IDs
    """
    try:
        # Validate parameters
        job_type = validate_job_search_params(job_type, job_type_mapping)
        employment_type = validate_job_search_params(employment_type, employment_type_mapping)
        experience_level = validate_job_search_params(experience, experience_type_mapping)
        
        # Try to use LinkedIn API if credentials are available
        linkedin_email = os.getenv("LINKEDIN_EMAIL")
        linkedin_pass = os.getenv("LINKEDIN_PASS")
        
        if linkedin_email and linkedin_pass:
            try:
                from linkedin_api import Linkedin
                
                api = Linkedin(linkedin_email, linkedin_pass)
                job_postings = api.search_jobs(
                    keywords=keywords,
                    job_type=employment_type,
                    location_name=location_name,
                    remote=job_type,
                    limit=limit,
                    experience=experience_level,
                    listed_at=listed_at,
                    distance=distance,
                )
                
                # Extract job IDs from tracking URNs
                job_ids = []
                for job in job_postings:
                    if "trackingUrn" in job:
                        job_id = job["trackingUrn"].split("jobPosting:")[-1]
                        job_ids.append(job_id)
                
                logger.info(f"Retrieved {len(job_ids)} job IDs via LinkedIn API")
                return job_ids
                
            except ImportError:
                logger.warning("linkedin-api package not available, falling back to web scraping")
            except Exception as api_error:
                logger.error(f"LinkedIn API error: {str(api_error)}")
        
        # Fallback to web scraping if API is not available
        return get_job_ids_via_scraping(keywords, location_name, employment_type, limit, job_type, experience)
        
    except Exception as e:
        logger.error(f"Error getting job IDs from LinkedIn API: {str(e)}")
        return []


def get_job_ids_via_scraping(
    keywords: str,
    location_name: str = None,
    employment_type: List[str] = None,
    limit: int = 10,
    job_type: List[str] = None,
    experience: List[str] = None,
) -> List[str]:
    """
    Get job IDs via web scraping as fallback method.
    
    Args:
        keywords: Search keywords
        location_name: Geographic location
        employment_type: Employment type filters
        limit: Maximum number of jobs
        job_type: Job type filters
        experience: Experience level filters
        
    Returns:
        List of job IDs
    """
    try:
        # Build search URL
        job_url = build_linkedin_job_url(
            keywords=keywords,
            location=location_name,
            employment_type=employment_type,
            experience_level=experience,
            job_type=job_type,
        )

        # Set up headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        # Add random delay to avoid being blocked
        time.sleep(random.uniform(1, 3))

        # Send request with timeout
        response = requests.get(job_url, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse HTML response
        soup = BeautifulSoup(response.content, "html.parser")
        job_cards = soup.find_all("li")

        job_ids = []
        for job_card in job_cards:
            try:
                base_card_div = job_card.find("div", {"class": "base-card"})
                if base_card_div and base_card_div.get("data-entity-urn"):
                    job_id = base_card_div.get("data-entity-urn").split(":")[-1]
                    job_ids.append(job_id)
                    
                    if len(job_ids) >= limit:
                        break
                        
            except Exception as card_error:
                logger.warning(f"Error parsing job card: {str(card_error)}")
                continue

        logger.info(f"Scraped {len(job_ids)} job IDs for '{keywords}'")
        return job_ids
        
    except requests.RequestException as req_error:
        logger.error(f"Request error during scraping: {str(req_error)}")
        return []
    except Exception as e:
        logger.error(f"Error scraping job IDs: {str(e)}")
        return []


def get_job_ids(
    keywords: str,
    location_name: str = None,
    employment_type: Optional[List[Literal["full-time", "contract", "part-time", "temporary", "internship", "volunteer", "other"]]] = None,
    limit: Optional[int] = 10,
    job_type: Optional[List[Literal["onsite", "remote", "hybrid"]]] = None,
    experience: Optional[List[Literal["internship", "entry-level", "associate", "mid-senior-level", "director", "executive"]]] = None,
    listed_at: Optional[Union[int, str]] = 86400,
    distance: Optional[Union[int, str]] = 25,
) -> List[str]:
    """
    Main function to get job IDs with multiple fallback methods.
    
    Args:
        keywords: Search keywords
        location_name: Geographic location
        employment_type: Employment type preferences
        limit: Maximum number of jobs to retrieve
        job_type: Job type preferences (remote/onsite/hybrid)
        experience: Experience level preferences
        listed_at: Time since posting in seconds
        distance: Search radius
        
    Returns:
        List of job IDs
    """
    try:
        # Convert string parameters to integers
        if isinstance(listed_at, str):
            listed_at = int(listed_at)
        if isinstance(distance, str):
            distance = int(distance)
        if isinstance(limit, str):
            limit = int(limit)
            
        # Ensure reasonable limits
        limit = min(max(limit, 1), 50)  # Between 1 and 50
        
        # Determine which method to use
        use_api = os.getenv("LINKEDIN_SEARCH") == "linkedin_api"
        
        if use_api:
            job_ids = get_job_ids_from_linkedin_api(
                keywords=keywords,
                location_name=location_name,
                employment_type=employment_type,
                limit=limit,
                job_type=job_type,
                experience=experience,
                listed_at=listed_at,
                distance=distance,
            )
        else:
            job_ids = get_job_ids_via_scraping(
                keywords=keywords,
                location_name=location_name,
                employment_type=employment_type,
                limit=limit,
                job_type=job_type,
                experience=experience,
            )
        
        # If no results, try alternative search terms
        if not job_ids and keywords:
            logger.info("No results found, trying alternative search...")
            
            # Try broader search terms
            alternative_keywords = keywords.split()[0] if ' ' in keywords else keywords
            job_ids = get_job_ids_via_scraping(
                keywords=alternative_keywords,
                location_name=location_name,
                limit=min(limit, 5)  # Reduce limit for alternative search
            )
        
        logger.info(f"Final result: {len(job_ids)} job IDs retrieved for '{keywords}'")
        return job_ids
        
    except Exception as e:
        logger.error(f"Error in get_job_ids: {str(e)}")
        return []


async def fetch_job_details(session: aiohttp.ClientSession, job_id: str) -> Dict[str, Any]:
    """
    Fetch detailed information for a specific job ID.
    
    Args:
        session: aiohttp session
        job_id: LinkedIn job ID
        
    Returns:
        Dict containing job details
    """
    try:
        # Construct job details URL
        job_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
        
        # Set headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Add random delay
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        async with session.get(job_url, headers=headers, timeout=30) as response:
            if response.status != 200:
                logger.warning(f"HTTP {response.status} for job {job_id}")
                return create_empty_job_dict(job_id)
                
            html_content = await response.text()
            soup = BeautifulSoup(html_content, "html.parser")

            # Parse job information
            job_post = parse_job_details_from_html(soup, job_id)
            
            # Add metadata
            job_post["job_id"] = job_id
            job_post["scraped_at"] = datetime.now().isoformat()
            job_post["source"] = "linkedin"
            
            return job_post
            
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching job {job_id}")
        return create_empty_job_dict(job_id, error="timeout")
    except Exception as e:
        logger.error(f"Error fetching job {job_id}: {str(e)}")
        return create_empty_job_dict(job_id, error=str(e))


def parse_job_details_from_html(soup: BeautifulSoup, job_id: str) -> Dict[str, Any]:
    """
    Parse job details from HTML soup.
    
    Args:
        soup: BeautifulSoup object
        job_id: Job ID for reference
        
    Returns:
        Dict containing parsed job details
    """
    job_post = {}
    
    try:
        # Extract job title
        title_element = soup.find("h2", {
            "class": "top-card-layout__title font-sans text-lg papabear:text-xl font-bold leading-open text-color-text mb-0 topcard__title"
        })
        job_post["job_title"] = title_element.text.strip() if title_element else ""
    except:
        job_post["job_title"] = ""

    try:
        # Extract location
        location_element = soup.find("span", {"class": "topcard__flavor topcard__flavor--bullet"})
        job_post["job_location"] = location_element.text.strip() if location_element else ""
    except:
        job_post["job_location"] = ""

    try:
        # Extract company name
        company_element = soup.find("a", {"class": "topcard__org-name-link topcard__flavor--black-link"})
        job_post["company_name"] = company_element.text.strip() if company_element else ""
    except:
        job_post["company_name"] = ""

    try:
        # Extract posting time
        time_element = soup.find("span", {"class": "posted-time-ago__text topcard__flavor--metadata"})
        job_post["time_posted"] = time_element.text.strip() if time_element else ""
    except:
        job_post["time_posted"] = ""

    try:
        # Extract number of applicants
        applicants_element = soup.find("span", {
            "class": "num-applicants__caption topcard__flavor--metadata topcard__flavor--bullet"
        })
        job_post["num_applicants"] = applicants_element.text.strip() if applicants_element else ""
    except:
        job_post["num_applicants"] = ""

    try:
        # Extract job description
        description_element = soup.find("div", {"class": "decorated-job-posting__details"})
        if description_element:
            job_post["job_desc_text"] = description_element.text.strip()
        else:
            # Try alternative description selectors
            description_element = soup.find("div", {"class": "description__text"})
            job_post["job_desc_text"] = description_element.text.strip() if description_element else ""
    except:
        job_post["job_desc_text"] = ""

    try:
        # Extract apply link
        apply_link_element = soup.find("a", class_="topcard__link")
        if apply_link_element:
            apply_link = apply_link_element.get("href")
            job_post["apply_link"] = apply_link
        else:
            job_post["apply_link"] = f"https://www.linkedin.com/jobs/view/{job_id}"
    except:
        job_post["apply_link"] = f"https://www.linkedin.com/jobs/view/{job_id}"

    try:
        # Extract employment type if available
        employment_element = soup.find("span", {"class": "description__job-criteria-text"})
        job_post["employment_type"] = employment_element.text.strip() if employment_element else ""
    except:
        job_post["employment_type"] = ""

    try:
        # Extract experience level if available
        criteria_items = soup.find_all("span", {"class": "description__job-criteria-text"})
        job_post["experience_level"] = ""
        if len(criteria_items) > 1:
            job_post["experience_level"] = criteria_items[1].text.strip()
    except:
        job_post["experience_level"] = ""

    # Add additional metadata
    job_post["remote_option"] = detect_remote_work(job_post.get("job_desc_text", "") + job_post.get("job_location", ""))
    job_post["salary_range"] = extract_salary_info(job_post.get("job_desc_text", ""))
    
    return job_post


async def get_job_details_from_linkedin_api(job_id: str) -> Dict[str, Any]:
    """
    Get job details using LinkedIn API (when available).
    
    Args:
        job_id: LinkedIn job ID
        
    Returns:
        Dict containing job details
    """
    try:
        linkedin_email = os.getenv("LINKEDIN_EMAIL")
        linkedin_pass = os.getenv("LINKEDIN_PASS")
        
        if not linkedin_email or not linkedin_pass:
            return create_empty_job_dict(job_id, error="LinkedIn credentials not available")
            
        from linkedin_api import Linkedin
        
        api = Linkedin(linkedin_email, linkedin_pass)
        
        # Use sync_to_async to make the blocking call async
        job_data = await sync_to_async(api.get_job)(job_id)
        
        if not job_data:
            return create_empty_job_dict(job_id, error="No data returned from API")

        # Parse API response into standardized format
        job_data_dict = {
            "job_id": job_id,
            "company_name": extract_nested_value(job_data, ["companyDetails", "com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany", "companyResolutionResult", "name"], ""),
            "company_url": extract_nested_value(job_data, ["companyDetails", "com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany", "companyResolutionResult", "url"], ""),
            "job_desc_text": extract_nested_value(job_data, ["description", "text"], ""),
            "work_remote_allowed": job_data.get("workRemoteAllowed", False),
            "job_title": job_data.get("title", ""),
            "apply_link": extract_nested_value(job_data, ["applyMethod", "com.linkedin.voyager.jobs.OffsiteApply", "companyApplyUrl"], f"https://www.linkedin.com/jobs/view/{job_id}"),
            "job_location": job_data.get("formattedLocation", ""),
            "employment_type": job_data.get("employmentStatus", ""),
            "experience_level": job_data.get("experienceLevel", ""),
            "posted_at": job_data.get("listedAt", ""),
            "num_applicants": job_data.get("numApplicants", ""),
            "scraped_at": datetime.now().isoformat(),
            "source": "linkedin_api"
        }
        
        # Add additional processing
        job_data_dict["remote_option"] = job_data_dict.get("work_remote_allowed", False)
        job_data_dict["salary_range"] = extract_salary_info(job_data_dict.get("job_desc_text", ""))
        
        return job_data_dict
        
    except ImportError:
        logger.error("linkedin-api package not available")
        return create_empty_job_dict(job_id, error="LinkedIn API not available")
    except Exception as e:
        logger.error(f"Error getting job details from LinkedIn API: {str(e)}")
        return create_empty_job_dict(job_id, error=str(e))


async def fetch_all_jobs(job_ids: List[str], batch_size: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch details for all job IDs with batching and error handling.
    
    Args:
        job_ids: List of job IDs to fetch
        batch_size: Number of concurrent requests
        
    Returns:
        List of job detail dictionaries
    """
    if not job_ids:
        logger.warning("No job IDs provided to fetch_all_jobs")
        return []
    
    try:
        # Check if we should use LinkedIn API
        use_api = os.getenv("LINKEDIN_SEARCH") == "linkedin_api"
        
        if use_api:
            # Use LinkedIn API
            tasks = [get_job_details_from_linkedin_api(job_id) for job_id in job_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Use web scraping with session
            connector = aiohttp.TCPConnector(limit=batch_size, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                # Process in batches to avoid overwhelming the server
                all_results = []
                
                for i in range(0, len(job_ids), batch_size):
                    batch = job_ids[i:i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} jobs")
                    
                    # Create tasks for current batch
                    batch_tasks = [fetch_job_details(session, job_id) for job_id in batch]
                    
                    # Execute batch with error handling
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    all_results.extend(batch_results)
                    
                    # Add delay between batches
                    if i + batch_size < len(job_ids):
                        await asyncio.sleep(2)
                
                results = all_results
        
        # Process results and handle exceptions
        processed_results = []
        successful_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing job {job_ids[i] if i < len(job_ids) else 'unknown'}: {str(result)}")
                processed_results.append(create_empty_job_dict(job_ids[i] if i < len(job_ids) else "unknown", error=str(result)))
            elif result and isinstance(result, dict):
                processed_results.append(result)
                successful_count += 1
            else:
                logger.warning(f"Empty result for job {job_ids[i] if i < len(job_ids) else 'unknown'}")
                processed_results.append(create_empty_job_dict(job_ids[i] if i < len(job_ids) else "unknown", error="Empty result"))
        
        logger.info(f"Successfully fetched {successful_count}/{len(job_ids)} job details")
        return processed_results
        
    except Exception as e:
        logger.error(f"Error in fetch_all_jobs: {str(e)}")
        # Return empty job dicts for all IDs if there's a major error
        return [create_empty_job_dict(job_id, error=str(e)) for job_id in job_ids]


def create_empty_job_dict(job_id: str, error: str = None) -> Dict[str, Any]:
    """
    Create an empty job dictionary with default values.
    
    Args:
        job_id: Job ID
        error: Error message if applicable
        
    Returns:
        Dict with empty job data
    """
    empty_job = {
        "job_id": job_id,
        "job_title": "",
        "company_name": "",
        "job_location": "",
        "job_desc_text": "",
        "apply_link": f"https://www.linkedin.com/jobs/view/{job_id}",
        "time_posted": "",
        "num_applicants": "",
        "employment_type": "",
        "experience_level": "",
        "remote_option": False,
        "salary_range": "",
        "scraped_at": datetime.now().isoformat(),
        "source": "fallback"
    }
    
    if error:
        empty_job["error"] = error
        
    return empty_job


def extract_nested_value(data: dict, keys: List[str], default: Any = None) -> Any:
    """
    Safely extract nested values from dictionary.
    
    Args:
        data: Dictionary to extract from
        keys: List of keys to traverse
        default: Default value if extraction fails
        
    Returns:
        Extracted value or default
    """
    try:
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    except:
        return default


def detect_remote_work(text: str) -> bool:
    """
    Detect if a job offers remote work based on text content.
    
    Args:
        text: Job description or location text
        
    Returns:
        Boolean indicating remote work availability
    """
    if not text:
        return False
        
    text_lower = text.lower()
    remote_indicators = [
        "remote", "work from home", "wfh", "telecommute", "distributed",
        "anywhere", "virtual", "home office", "remote-friendly"
    ]
    
    return any(indicator in text_lower for indicator in remote_indicators)


def extract_salary_info(text: str) -> str:
    """
    Extract salary information from job description text.
    
    Args:
        text: Job description text
        
    Returns:
        Extracted salary information or empty string
    """
    if not text:
        return ""
    
    import re
    
    # Common salary patterns
    salary_patterns = [
        r'\$[\d,]+\s*-\s*\$[\d,]+',  # $50,000 - $70,000
        r'\$[\d,]+\s*to\s*\$[\d,]+',  # $50,000 to $70,000
        r'\$[\d,]+k?\s*-\s*\$?[\d,]+k?',  # $50k - $70k
        r'[\d,]+\s*-\s*[\d,]+\s*(?:USD|dollars?)',  # 50,000 - 70,000 USD
        r'salary:?\s*\$?[\d,]+(?:\s*-\s*\$?[\d,]+)?',  # Salary: $50,000
    ]
    
    for pattern in salary_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group().strip()
    
    return ""


def enhance_job_data(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance job data with additional processing and insights.
    
    Args:
        job_data: Raw job data dictionary
        
    Returns:
        Enhanced job data dictionary
    """
    try:
        # Add job URL if not present
        if not job_data.get("apply_link") and job_data.get("job_id"):
            job_data["apply_link"] = f"https://www.linkedin.com/jobs/view/{job_data['job_id']}"
        
        # Standardize remote work detection
        desc_text = job_data.get("job_desc_text", "")
        location_text = job_data.get("job_location", "")
        job_data["remote_option"] = detect_remote_work(desc_text + " " + location_text)
        
        # Extract and standardize salary information
        if not job_data.get("salary_range"):
            job_data["salary_range"] = extract_salary_info(desc_text)
        
        # Add relevance score placeholder
        job_data["relevance_score"] = 0.8  # Default relevance
        
        # Add data quality score
        quality_factors = [
            bool(job_data.get("job_title")),
            bool(job_data.get("company_name")),
            bool(job_data.get("job_desc_text")),
            bool(job_data.get("job_location")),
            bool(job_data.get("apply_link"))
        ]
        job_data["data_quality_score"] = sum(quality_factors) / len(quality_factors)
        
        # Add enhanced metadata
        job_data["enhanced_at"] = datetime.now().isoformat()
        
        return job_data
        
    except Exception as e:
        logger.error(f"Error enhancing job data: {str(e)}")
        return job_data


# Utility functions for backward compatibility and testing
def test_job_search_functionality():
    """Test the job search functionality with sample data."""
    try:
        # Test job ID retrieval
        test_keywords = "software engineer"
        test_location = "San Francisco, CA"
        
        logger.info("Testing job search functionality...")
        
        job_ids = get_job_ids(
            keywords=test_keywords,
            location_name=test_location,
            limit=3
        )
        
        if job_ids:
            logger.info(f"✅ Successfully retrieved {len(job_ids)} job IDs")
            
            # Test job details fetching
            job_details = asyncio.run(fetch_all_jobs(job_ids[:2]))  # Test with first 2 jobs
            
            if job_details:
                logger.info(f"✅ Successfully fetched details for {len(job_details)} jobs")
                return True
            else:
                logger.warning("⚠️ No job details retrieved")
                return False
        else:
            logger.warning("⚠️ No job IDs retrieved")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False


def get_search_statistics() -> Dict[str, Any]:
    """Get statistics about search functionality and configuration."""
    return {
        "linkedin_api_available": bool(os.getenv("LINKEDIN_EMAIL") and os.getenv("LINKEDIN_PASS")),
        "linkedin_search_method": os.getenv("LINKEDIN_SEARCH", "scraping"),
        "supported_employment_types": list(employment_type_mapping.keys()),
        "supported_experience_levels": list(experience_type_mapping.keys()),
        "supported_job_types": list(job_type_mapping.keys()),
        "default_search_limit": 10,
        "max_search_limit": 50,
        "timestamp": datetime.now().isoformat()
    }


# Export main functions
__all__ = [
    'get_job_ids',
    'fetch_all_jobs',
    'build_linkedin_job_url',
    'test_job_search_functionality',
    'get_search_statistics'
]