from typing import List
from crewai import Agent
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool
from tools.youtube_search_tools import YoutubeVideoSearchTool
from pytrends.request import TrendReq
from pydantic import BaseModel, Field

class CompanyResearchAgents():

    def __init__(self):
        self.searchInternetTool = SerperDevTool()
        self.youtubeSearchTool = YoutubeVideoSearchTool()
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")

    def research_manager(self, companies: List[str], positions: List[str]) -> Agent:
        return Agent(
            role="Company Research Manager",
            goal=f"""Generate a list of JSON objects containing the urls for 3 recent blog articles and 
                the url and title for 3 recent YouTube interview, for each position in each company.
             
                Companies: {companies}
                Positions: {positions}

                Important:
                - The final list of JSON objects must include all companies and positions. Do not leave any out.
                - If you can't find information for a specific position, fill in the information with the word "MISSING".
                - Do not generate fake information. Only return the information you find. Nothing else!
                - Do not stop researching until you find the requested information for each position in each company.
                - All the companies and positions exist so keep researching until you find the information for each one.
                - Make sure you each researched position for each company contains 3 blog articles and 3 YouTube interviews.
                """,
            backstory="""As a Company Research Manager, you are responsible for aggregating all the researched information
                into a list.""",
            llm=self.llm,
            tools=[self.searchInternetTool, self.youtubeSearchTool],
            verbose=True,
            allow_delegation=True
        )

    def company_research_agent(self) -> Agent:
        return Agent(
            role="Company Research Agent",
            goal="""Look up the specific positions for a given company and find urls for 3 recent blog articles and 
                the url and title for 3 recent YouTube interview for each person in the specified positions. It is your job to return this collected 
                information in a JSON object""",
            backstory="""As a Company Research Agent, you are responsible for looking up specific positions 
                within a company and gathering relevant information.
                
                Important:
                - Once you've found the information, immediately stop searching for additional information.
                - Only return the requested information. NOTHING ELSE!
                - Make sure you find the persons name who holds the position.
                - Do not generate fake information. Only return the information you find. Nothing else!
                """,
            tools=[self.searchInternetTool, self.youtubeSearchTool],
            llm=self.llm,
            verbose=True
        )

class GoogleTrendsInput(BaseModel):
    keyword: str = Field(..., description="The keyword or topic to get trends for")
    timeframe: str = Field(default="today 3-m", description="Timeframe for the trend data. Default is 'today 3-m' (last 3 months)")

class GoogleTrendsTool(BaseTool):
    name = "google_trends"
    description = "Use this tool to get trending topics and their popularity from Google Trends."
    args_schema = GoogleTrendsInput

    def _run(self, keyword: str, timeframe: str = "today 3-m"):
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], timeframe=timeframe)
        
        # Get related queries
        related_queries = pytrends.related_queries()
        top_queries = related_queries[keyword]['top']
        
        if top_queries is not None and not top_queries.empty:
            return top_queries.head(5).to_dict()
        else:
            return f"No trending data found for {keyword}"

    async def _arun(self, keyword: str, timeframe: str = "today 3-m"):
        # For async operations. We'll just call the sync version for now.
        return self._run(keyword, timeframe)

class ContentGenerationAgents:
    def __init__(self):
        self.searchInternetTool = SerperDevTool()
        self.youtubeSearchTool = YoutubeVideoSearchTool()
        self.googleTrendsTool = GoogleTrendsTool()
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")

    def trend_analyzer(self) -> Agent:
        return Agent(
            role="Trend Analyzer",
            goal="Identify current trending topics within the specified industry",
            backstory="You are an expert at analyzing market trends and identifying hot topics.",
            tools=[self.googleTrendsTool, self.searchInternetTool],
            llm=self.llm,
            verbose=True
        )

    def research_coordinator(self) -> Agent:
        return Agent(
            role="Research Coordinator",
            goal="Gather relevant information from selected sources on the identified topic",
            backstory="You excel at finding and compiling information from various online sources.",
            tools=[self.searchInternetTool, self.youtubeSearchTool],
            llm=self.llm,
            verbose=True
        )

    def content_analyzer(self) -> Agent:
        return Agent(
            role="Content Analyzer",
            goal="Summarize and extract key information from the aggregated content",
            backstory="You are skilled at distilling complex information into clear, concise summaries.",
            tools=[],  # Add text summarization tools if needed
            llm=self.llm,
            verbose=True
        )

    def content_creator(self) -> Agent:
        return Agent(
            role="Content Creator",
            goal="Generate original content based on the research and user inputs",
            backstory="You are a talented writer capable of creating engaging content in various formats.",
            tools=[],  # The LLM itself will be the primary tool
            llm=self.llm,
            verbose=True
        )

    def fact_checker(self) -> Agent:
        return Agent(
            role="Fact Checker",
            goal="Ensure accuracy of the generated content and provide source attribution",
            backstory="You have a keen eye for detail and are committed to maintaining high standards of accuracy.",
            tools=[self.searchInternetTool],
            llm=self.llm,
            verbose=True
        )