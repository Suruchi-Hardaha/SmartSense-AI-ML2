# agents/report_generation.py - Report Generation Agent

from typing import Dict, Any, Optional, List
from agents.base import BaseAgent, AgentResponse, AgentType
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path

class ReportGenerationAgent(BaseAgent):
    """
    Report Generation Agent - Generates detailed reports with graphs
    Creates downloadable PDF reports with visualizations
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("ReportGeneration", AgentType.REPORT_GENERATION, config)
        
        # Report templates
        self.templates = {
            "property_analysis": self._generate_property_analysis_report,
            "market_research": self._generate_market_research_report,
            "comparison": self._generate_comparison_report,
            "renovation": self._generate_renovation_report,
            "investment": self._generate_investment_report
        }
        
        # Output directory
        self.output_dir = Path(config.get("output_dir", "reports")) if config else Path("reports")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Styling
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the report"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a472a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2c5530'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))
    
    async def execute(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        """
        Generate report based on input data and template type
        """
        try:
            report_type = input_data.get("report_type", "property_analysis")
            data = input_data.get("data", {})
            format = input_data.get("format", "pdf")
            
            # Select appropriate template
            if report_type in self.templates:
                report_path = await self.templates[report_type](data, context)
            else:
                report_path = await self._generate_generic_report(data, context)
            
            # Generate visualizations if data available
            visualizations = await self._generate_visualizations(data)
            
            return AgentResponse(
                success=True,
                data={
                    "report_path": str(report_path),
                    "report_type": report_type,
                    "format": format,
                    "visualizations": visualizations,
                    "download_link": f"/download-report/{report_path.name}"
                },
                message=f"Report generated successfully: {report_path.name}",
                metadata={
                    "pages": self._estimate_pages(data),
                    "generated_at": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=None,
                message=f"Report generation failed: {str(e)}"
            )
    
    async def _generate_property_analysis_report(self, data: Dict, context: Optional[Dict]) -> Path:
        """Generate comprehensive property analysis report"""
        filename = f"property_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = self.output_dir / filename
        
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        story = []
        
        # Title
        story.append(Paragraph("Property Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
        summary_text = self._create_executive_summary(data)
        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.2*inch))
        
        # Property Details Section
        if "properties" in data:
            story.append(Paragraph("Property Details", self.styles['CustomHeading']))
            for prop in data["properties"][:5]:  # Limit to 5 properties
                story.extend(self._create_property_section(prop))
            story.append(PageBreak())
        
        # Market Analysis Section
        if "market_data" in data:
            story.append(Paragraph("Market Analysis", self.styles['CustomHeading']))
            story.extend(self._create_market_analysis_section(data["market_data"]))
            story.append(Spacer(1, 0.2*inch))
        
        # Location Analysis
        if "neighborhood" in data:
            story.append(Paragraph("Location Analysis", self.styles['CustomHeading']))
            story.extend(self._create_location_section(data["neighborhood"]))
            story.append(PageBreak())
        
        # Price Comparison Chart
        if "properties" in data and len(data["properties"]) > 1:
            story.append(Paragraph("Price Comparison", self.styles['CustomHeading']))
            chart = self._create_price_comparison_chart(data["properties"])
            story.append(chart)
            story.append(Spacer(1, 0.2*inch))
        
        # Investment Recommendations
        story.append(Paragraph("Investment Recommendations", self.styles['CustomHeading']))
        recommendations = self._generate_recommendations(data)
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", self.styles['CustomBody']))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer_text = f"Report generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}"
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return filepath
    
    async def _generate_market_research_report(self, data: Dict, context: Optional[Dict]) -> Path:
        """Generate market research report"""
        filename = f"market_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = self.output_dir / filename
        
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        story = []
        
        # Title
        story.append(Paragraph("Market Research Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Market Overview
        story.append(Paragraph("Market Overview", self.styles['CustomHeading']))
        if "market_data" in data:
            overview = f"""The real estate market in {data.get('location', 'the area')} shows 
            {data['market_data'].get('trend', 'stable')} trends with an average price per sq.ft of 
            ₹{data['market_data'].get('avg_price_sqft', 'N/A')}."""
            story.append(Paragraph(overview, self.styles['CustomBody']))
        
        # Price Trends
        story.append(Paragraph("Price Trends", self.styles['CustomHeading']))
        if "trends" in data:
            trend_chart = self._create_trend_chart(data["trends"])
            story.append(trend_chart)
        
        # Supply-Demand Analysis
        story.append(Paragraph("Supply-Demand Analysis", self.styles['CustomHeading']))
        if "supply_demand" in data:
            supply_demand_table = self._create_supply_demand_table(data["supply_demand"])
            story.append(supply_demand_table)
        
        # Build PDF
        doc.build(story)
        return filepath
    
    async def _generate_comparison_report(self, data: Dict, context: Optional[Dict]) -> Path:
        """Generate property comparison report"""
        filename = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = self.output_dir / filename
        
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        story = []
        
        story.append(Paragraph("Property Comparison Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Comparison Table
        if "properties" in data and len(data["properties"]) > 1:
            comparison_table = self._create_comparison_table(data["properties"])
            story.append(comparison_table)
        
        doc.build(story)
        return filepath
    
    async def _generate_renovation_report(self, data: Dict, context: Optional[Dict]) -> Path:
        """Generate renovation cost estimation report"""
        filename = f"renovation_estimate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = self.output_dir / filename
        
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        story = []
        
        story.append(Paragraph("Renovation Cost Estimation Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Cost Breakdown
        if "cost_breakdown" in data:
            story.append(Paragraph("Cost Breakdown", self.styles['CustomHeading']))
            cost_table = self._create_cost_breakdown_table(data["cost_breakdown"])
            story.append(cost_table)
        
        # Timeline
        if "timeline" in data:
            story.append(Paragraph("Renovation Timeline", self.styles['CustomHeading']))
            timeline_text = f"Estimated completion time: {data['timeline']} weeks"
            story.append(Paragraph(timeline_text, self.styles['CustomBody']))
        
        doc.build(story)
        return filepath
    
    async def _generate_investment_report(self, data: Dict, context: Optional[Dict]) -> Path:
        """Generate investment analysis report"""
        filename = f"investment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = self.output_dir / filename
        
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        story = []
        
        story.append(Paragraph("Investment Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # ROI Analysis
        if "roi_analysis" in data:
            story.append(Paragraph("Return on Investment Analysis", self.styles['CustomHeading']))
            roi_chart = self._create_roi_chart(data["roi_analysis"])
            story.append(roi_chart)
        
        doc.build(story)
        return filepath
    
    async def _generate_generic_report(self, data: Dict, context: Optional[Dict]) -> Path:
        """Generate generic report for unspecified report types"""
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = self.output_dir / filename
        
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        story = []
        
        story.append(Paragraph("Real Estate Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Add all available data sections
        for key, value in data.items():
            if isinstance(value, (dict, list)) and value:
                story.append(Paragraph(key.replace("_", " ").title(), self.styles['CustomHeading']))
                
                if isinstance(value, dict):
                    for k, v in value.items():
                        story.append(Paragraph(f"{k}: {v}", self.styles['CustomBody']))
                elif isinstance(value, list):
                    for item in value[:5]:  # Limit items
                        story.append(Paragraph(f"• {item}", self.styles['CustomBody']))
                
                story.append(Spacer(1, 0.1*inch))
        
        doc.build(story)
        return filepath
    
    def _create_property_section(self, property_data: Dict) -> List:
        """Create a property details section"""
        elements = []
        
        # Property title
        title = property_data.get("title", "Property")
        elements.append(Paragraph(f"<b>{title}</b>", self.styles['Heading2']))
        
        # Property details table
        data = [
            ["Location", property_data.get("location", "N/A")],
            ["Price", f"₹{property_data.get('price', 0):,}"],
            ["Type", property_data.get("property_type", "N/A")],
            ["Size", f"{property_data.get('size', 'N/A')} sq.ft"],
        ]
        
        table = Table(data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_comparison_table(self, properties: List[Dict]) -> Table:
        """Create property comparison table"""
        headers = ["Feature", "Property 1", "Property 2", "Property 3"][:1 + min(len(properties), 3)]
        
        data = [headers]
        
        # Add comparison rows
        features = ["Price", "Location", "Size", "Rooms", "Bathrooms"]
        for feature in features:
            row = [feature]
            for prop in properties[:3]:
                if feature == "Price":
                    row.append(f"₹{prop.get('price', 0):,}")
                else:
                    row.append(str(prop.get(feature.lower(), "N/A")))
            data.append(row)
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        return table
    
    def _create_price_comparison_chart(self, properties: List[Dict]) -> Drawing:
        """Create price comparison bar chart"""
        drawing = Drawing(400, 200)
        
        bc = VerticalBarChart()
        bc.x = 50
        bc.y = 50
        bc.height = 125
        bc.width = 300
        
        # Data
        data = [[prop.get("price", 0) / 100000 for prop in properties[:5]]]  # Convert to lakhs
        bc.data = data
        
        # Labels
        bc.categoryAxis.categoryNames = [prop.get("title", f"Property {i+1}")[:15] 
                                        for i, prop in enumerate(properties[:5])]
        
        bc.bars[(0,0)].fillColor = colors.blue
        
        drawing.add(bc)
        
        return drawing
    
    def _create_trend_chart(self, trend_data: Dict) -> Drawing:
        """Create trend line chart"""
        drawing = Drawing(400, 200)
        
        lp = LinePlot()
        lp.x = 50
        lp.y = 50
        lp.height = 125
        lp.width = 300
        
        # Sample data (replace with actual trend data)
        data = [
            [(i, 5000 + i*100) for i in range(12)]  # Monthly price trend
        ]
        lp.data = data
        
        lp.lines[0].strokeColor = colors.blue
        
        drawing.add(lp)
        
        return drawing
    
    def _create_supply_demand_table(self, supply_demand_data: Dict) -> Table:
        """Create supply-demand analysis table"""
        data = [
            ["Metric", "Value"],
            ["New Launches", supply_demand_data.get("new_launches", "N/A")],
            ["Units Sold", supply_demand_data.get("units_sold", "N/A")],
            ["Inventory (months)", supply_demand_data.get("inventory_months", "N/A")],
            ["Demand Level", supply_demand_data.get("demand_level", "N/A")]
        ]
        
        table = Table(data, colWidths=[2.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        return table
    
    def _create_cost_breakdown_table(self, cost_breakdown: Dict) -> Table:
        """Create renovation cost breakdown table"""
        data = [["Item", "Cost (₹)"]]
        
        total = 0
        for item, cost in cost_breakdown.items():
            data.append([item, f"₹{cost:,}"])
            total += cost
        
        data.append(["TOTAL", f"₹{total:,}"])
        
        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('BACKGROUND', (0, -1), (-1, -1), colors.yellow),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        return table
    
    def _create_roi_chart(self, roi_data: Dict) -> Drawing:
        """Create ROI analysis chart"""
        drawing = Drawing(400, 200)
        
        pie = Pie()
        pie.x = 100
        pie.y = 50
        pie.width = 200
        pie.height = 100
        
        # Sample ROI data
        pie.data = [30, 20, 15, 35]  # Replace with actual ROI breakdown
        pie.labels = ["Rental Income", "Appreciation", "Tax Benefits", "Other"]
        
        pie.slices[0].fillColor = colors.blue
        pie.slices[1].fillColor = colors.green
        pie.slices[2].fillColor = colors.yellow
        pie.slices[3].fillColor = colors.red
        
        drawing.add(pie)
        
        return drawing
    
    def _create_executive_summary(self, data: Dict) -> str:
        """Create executive summary text"""
        summary = f"""This report provides a comprehensive analysis of real estate opportunities 
        in {data.get('location', 'the selected area')}. Based on current market conditions and 
        available data, the analysis covers {len(data.get('properties', []))} properties with 
        prices ranging from ₹{data.get('min_price', 'N/A')} to ₹{data.get('max_price', 'N/A')}. 
        The market shows {data.get('market_trend', 'stable')} trends with an average appreciation 
        of {data.get('appreciation', 'N/A')}% annually."""
        
        return summary
    
    def _create_market_analysis_section(self, market_data: Dict) -> List:
        """Create market analysis section"""
        elements = []
        
        analysis_text = f"""The current market analysis reveals:
        
        • Average Price per Sq.Ft: ₹{market_data.get('avg_price_sqft', 'N/A')}
        • Monthly Change: {market_data.get('monthly_change', 'N/A')}%
        • Yearly Change: {market_data.get('yearly_change', 'N/A')}%
        • Market Sentiment: {market_data.get('sentiment', 'Neutral')}
        """
        
        elements.append(Paragraph(analysis_text, self.styles['CustomBody']))
        
        return elements
    
    def _create_location_section(self, neighborhood_data: Dict) -> List:
        """Create location analysis section"""
        elements = []
        
        location_text = f"""Location Overview:
        
        {neighborhood_data.get('overview', 'No overview available')}
        
        Key Connectivity:
        • Nearest Metro: {neighborhood_data.get('connectivity', {}).get('nearest_metro', 'N/A')}
        • Airport Distance: {neighborhood_data.get('connectivity', {}).get('airport_distance', 'N/A')}
        
        Amenities:
        • Schools: {neighborhood_data.get('amenities', {}).get('schools', 0)}
        • Hospitals: {neighborhood_data.get('amenities', {}).get('hospitals', 0)}
        • Shopping Centers: {neighborhood_data.get('amenities', {}).get('shopping_centers', 0)}
        """
        
        elements.append(Paragraph(location_text, self.styles['CustomBody']))
        
        return elements
    
    def _generate_recommendations(self, data: Dict) -> List[str]:
        """Generate investment recommendations based on data"""
        recommendations = []
        
        # Price-based recommendations
        if "properties" in data:
            avg_price = sum(p.get("price", 0) for p in data["properties"]) / len(data["properties"])
            if avg_price < 5000000:
                recommendations.append("Properties in this range offer good entry-level investment opportunities")
            elif avg_price > 10000000:
                recommendations.append("Premium properties suitable for long-term capital appreciation")
        
        # Market trend recommendations
        if "market_data" in data:
            if data["market_data"].get("trend") == "increasing":
                recommendations.append("Market shows positive growth trends, favorable for investment")
            elif data["market_data"].get("trend") == "stable":
                recommendations.append("Stable market conditions provide predictable returns")
        
        # Location recommendations
        if "neighborhood" in data:
            if data["neighborhood"].get("development_projects"):
                recommendations.append("Upcoming infrastructure projects may boost property values")
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Conduct detailed property inspection before purchase",
                "Verify all legal documents and clearances",
                "Consider long-term appreciation potential",
                "Evaluate rental yield possibilities"
            ]
        
        return recommendations
    
    async def _generate_visualizations(self, data: Dict) -> List[str]:
        """Generate additional visualizations using matplotlib"""
        visualizations = []
        
        try:
            # Price distribution chart
            if "properties" in data and len(data["properties"]) > 2:
                plt.figure(figsize=(10, 6))
                prices = [p.get("price", 0) / 100000 for p in data["properties"]]  # In lakhs
                plt.hist(prices, bins=10, edgecolor='black')
                plt.xlabel("Price (Lakhs)")
                plt.ylabel("Number of Properties")
                plt.title("Property Price Distribution")
                
                chart_path = self.output_dir / f"price_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_path)
                plt.close()
                
                visualizations.append(str(chart_path))
            
            # Location-wise comparison
            if "by_location" in data:
                plt.figure(figsize=(12, 6))
                locations = [item["location"] for item in data["by_location"]]
                counts = [item["count"] for item in data["by_location"]]
                
                plt.bar(locations, counts)
                plt.xlabel("Location")
                plt.ylabel("Number of Properties")
                plt.title("Properties by Location")
                plt.xticks(rotation=45, ha='right')
                
                chart_path = self.output_dir / f"location_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()
                
                visualizations.append(str(chart_path))
        
        except Exception as e:
            self.logger.error(f"Visualization generation error: {str(e)}")
        
        return visualizations
    
    def _estimate_pages(self, data: Dict) -> int:
        """Estimate number of pages in the report"""
        base_pages = 2  # Title and summary
        
        if "properties" in data:
            base_pages += min(len(data["properties"]), 5)  # 1 page per property (max 5)
        
        if "market_data" in data:
            base_pages += 2
        
        if "neighborhood" in data:
            base_pages += 1
        
        return base_pages