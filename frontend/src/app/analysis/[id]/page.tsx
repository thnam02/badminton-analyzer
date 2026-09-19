import { AnalysisResultView } from "@/components/analysis/AnalysisResultView";

type Props = {
  params: Promise<{ id: string }>;
};

export default async function AnalysisPage({ params }: Props) {
  const { id } = await params;
  return <AnalysisResultView analysisId={id} />;
}
